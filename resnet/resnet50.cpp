#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "common.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "opencv2/opencv.hpp"

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 64;
static const int INPUT_W = 64;
static const int OUTPUT_SIZE = 6;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;

static Logger gLogger;

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    std::cout << "len " << len << std::endl;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

IActivationLayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{1, 1}, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolution(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStride(DimsHW{stride, stride});
    conv2->setPadding(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IConvolutionLayer* conv3 = network->addConvolution(*relu2->getOutput(0), outch * 4, DimsHW{1, 1}, weightMap[lname + "conv3.weight"], emptywts);
    assert(conv3);

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "bn3", 1e-5);

    IElementWiseLayer* ew1;
    if (stride != 1 || inch != outch * 4) {
        IConvolutionLayer* conv4 = network->addConvolution(input, outch * 4, DimsHW{1, 1}, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(conv4);
        conv4->setStride(DimsHW{stride, stride});

        IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, DataType dt)
{
    INetworkDefinition* network = builder->createNetwork();

    // Create input tensor of shape { 1, 1, 32, 32 } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../resnet50_fc.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

	std::cout << "0\n";

    // Add convolution layer with 6 outputs and a 5x5 filter.
    IConvolutionLayer* conv1 = network->addConvolution(*data, 64, DimsHW{7, 7}, weightMap["conv1.weight"], emptywts);
	std::cout << "0.1\n";

    assert(conv1);
    conv1->setStride(DimsHW{2, 2});
    conv1->setPadding(DimsHW{3, 3});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer* pool1 = network->addPooling(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStride(DimsHW{2, 2});
    pool1->setPadding(DimsHW{1, 1});

	std::cout << "1\n";

    IActivationLayer* x = bottleneck(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "layer1.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "layer1.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "layer1.2.");
	std::cout << "2\n";

    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 128, 2, "layer2.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "layer2.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "layer2.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "layer2.3.");
	std::cout << "3\n";

    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 256, 2, "layer3.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.3.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.4.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.5.");
	std::cout << "4\n";

    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 512, 2, "layer4.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "layer4.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "layer4.2.");
	std::cout << "5\n";

#if 0
    IPoolingLayer* pool2 = network->addPooling(*x->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    assert(pool2);
    pool2->setStride(DimsHW{1, 1});

    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), 1000, weightMap["fc.weight"], weightMap["fc.bias"]);
    assert(fc1);

    fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*fc1->getOutput(0));
#else
	IPoolingLayer* pool2 = network->addPooling(*x->getOutput(0), PoolingType::kAVERAGE, DimsHW{ 2, 2 });
	assert(pool2);
	pool2->setStride(DimsHW{ 1, 1 });

	IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), OUTPUT_SIZE, weightMap["fc.weight"], weightMap["fc.bias"]);
	assert(fc1);

	fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
	std::cout << "set name out" << std::endl;
	network->markOutput(*fc1->getOutput(0));
	
	//for (int i = 0; i < 8; i++)
	//{
	//	std::cout << pool2->getOutput(0)->getDimensions().d[i] << std::endl;
	//}
	//std::cout << "custom1\n";
	//IFullyConnectedLayer* fc10 = network->addFullyConnected(*pool2->getOutput(0), 256, weightMap["fc.0.weight"], weightMap["fc.0.bias"]);
	//std::cout << "custom2\n";
	//assert(fc10);

	///*IActivationLayer* relu10 = network->addActivation(*fc10->getOutput(0), ActivationType::kRELU);
	//std::cout << "custom3\n";
	//assert(relu10);
	//IFullyConnectedLayer* fc11 = network->addFullyConnected(*relu10->getOutput(0), OUTPUT_SIZE, weightMap["fc.3.weight"], weightMap["fc.3.bias"]);
	//std::cout << "custom4\n";
	//assert(fc11);*/

	////ISoftMaxLayer* sm = network->addSoftMax(*fc11->getOutput(0));
	////std::cout << "custom5\n";
	////assert(sm);

	//fc10->getOutput(0)->setName(OUTPUT_BLOB_NAME);
	//std::cout << "set name out" << std::endl;
	//network->markOutput(*fc10->getOutput(0));
#endif

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
	std::cout << "build out1" << std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);

	std::cout << "10\n";

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, DataType::kFLOAT);
    assert(engine != nullptr);
	std::cout << "11\n";

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./resnet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./resnet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("resnet50.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("resnet50.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        return -1;
    }


    // Subtract mean from image
#if 0
    float data[3 * INPUT_H * INPUT_W];
    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        data[i] = 1.0;
#else
	//float data[3 * INPUT_H * INPUT_W];
	cv::Mat img = cv::imread("G:/aoi_resnet/test/signs_3/img_0092.png");
	img.convertTo(img, CV_32FC3, 1 / 255.0);

	float *img_data = (float *)img.data;
	//std::cout << "data0: " << data[0] << std::endl;

	/*int img_size = 3 * INPUT_H * INPUT_W;
	float data[3 * INPUT_H * INPUT_W];
	int count = 0;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < img_size; j+=3)
		{
			data[count] = img_data[j + i];
			count++;
		}
	}*/

#endif

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    float prob[OUTPUT_SIZE];
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::system_clock::now();

		int img_size = 3 * INPUT_H * INPUT_W;
		float data[3 * INPUT_H * INPUT_W];
		int count = 0;
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < img_size; j += 3)
			{
				data[count] = img_data[j + i];
				count++;
			}
		}

        doInference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    {
        std::cout << prob[i] << ", ";
    }
    std::cout << std::endl;
	/*for (unsigned int i = 0; i < 10; i++)
	{
		std::cout << prob[OUTPUT_SIZE - 10 + i] << ", ";
	}
	std::cout << std::endl;*/

    return 0;
}
