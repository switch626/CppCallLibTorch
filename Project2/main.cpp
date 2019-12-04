#include <torch/script.h> // One-stop header.
#include <opencv.hpp>
#include <iostream>
#include <memory>

#include "common.h"

int main(int argc, const char* argv[]) {
	if (argc < 4) {
		std::cerr << "usage: example-app <path-to-exported-script-module> <img1.jpg> <img2.jpg>\n";
		return -1;
	}

	//define cpu device for host processing data
	//创建一个cpu设备对象，后续有用
	torch::Device cpu_device(torch::kCPU);

	//detect GPU
	//查询gpu，并创建auto_device变量，用于gpu/cpu自动选择
	torch::DeviceType device_type;
	bool is_gpu = torch::cuda::is_available();

	if (is_gpu)
	{
		device_type = torch::kCUDA;
	}
	else
	{
		device_type = torch::kCPU;
	}
	torch::Device auto_device(device_type);

	// Deserialize the ScriptModule from a file using torch::jit::load().
	//反序列化pytorch模型，并加载到module变量
	//std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1], auto_device);
	torch::jit::script::Module module;
	try
	{
		module = torch::jit::load(argv[1], auto_device);

	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}
	//assert(module != nullptr);

	//start to do inference
	//prepare inputs date var
	//网络的inputs，这里是个固定类型的变量
	std::vector<torch::jit::IValue> inputs;

	//try to read a image to run the model
	//make the sequence in batch,channel,w,h
	//此处我们要使用opencv读取两张图片
	std::vector<cv::Mat> image_vec; //这里，我们会把读取到的两张图片放进这个vector中，方便后续调用
	std::vector<cv::Mat>& r_image_vec = image_vec;//上面vector的引用，方便当作函数参数使用
	std::vector<torch::Tensor> img_t;//我们要把opencv读取的mat转换到libtorch的tensor变量，这里创建vector，方便后续使用
	std::vector<torch::Tensor>& r_img_t = img_t;//tensor vector的引用，方便当作函数参数使用

	//读取图片，并放入image_vec中，方便后续使用
	readMatImg(argv[2], r_image_vec);
	//从image_vec中提取图片，并转换成tensor格式，存放入img_t的vector中，以便后面步骤使用
	prepareImgTensor(r_img_t, r_image_vec, 0);

	//重复上面步骤，读取第二张图片。因为我们后面要演示batch的input，所以使用了两张图片。如果你只有一张图片，没关系，vector中放入一个tensor就行了
	readMatImg(argv[3], r_image_vec);
	prepareImgTensor(r_img_t, r_image_vec, 1);

	//move to device
	//把tensor放入到device中，这里auto_device会根据系统的实际情况，选择是否放入gpu
	img_t[0] = img_t[0].to(auto_device);
	img_t[1] = img_t[1].to(auto_device);

	//我们是inference，所以要把tensor的autograd关掉，不然浪费显存资源
	auto img_var = torch::autograd::make_variable(img_t[0], false);
	auto img_var1 = torch::autograd::make_variable(img_t[1], false);

	//上面我们提到过，我们使用了两张图，所以呢，我们要把两个tensor拼接成如下形式
	// [[1,3,224,224],
	//  [1,3,224,224]]
	//这样系统就能并行处理两张图片了
	auto img_vec = torch::cat({ img_var,img_var1 }, 0);
	//cat后的tensor，要push到inputs的变量中
	inputs.push_back(img_vec);
	std::cout << "inputs:" << inputs.size() << std::endl;

	std::cout << "start..." << std::endl;
	//开始调用模型进行inference，并获取结果out_tensor
	torch::Tensor out_tensor = module.forward(inputs).toTensor();
	//std::cout<<"end..."<<"out tensor size:"<<out_tensor.sizes()<<std::endl;
	//由于此前我们用pytorch导出模型的时候，最后一层模型的输出是log_softmax，所以我们这里要转一下，变成softmax
	out_tensor = torch::softmax(out_tensor, 1);
	//记得把结果提取到cpu端来处理后续步骤
	out_tensor = out_tensor.to(cpu_device);
	//我们来定一个out_tensor的引用，给后面处理推理结果使用
	torch::Tensor& r_out_tensor = out_tensor;
	//我们要从out_tensor里面取出我们要的分类index和confidence,详情请参看getInferenceResults的函数说明
	std::vector<std::tuple<int, float>> ret = getInferenceResults(r_out_tensor);

	for (auto& r : ret)
	{
		std::cout << "id:" << std::get<0>(r) << " conf:" << std::get<1>(r) << std::endl;
	}

	return 0;

}

//std::string image_path = "H:\\项目\\BGSS\\sortsout\\Progress\\recordLog\\pictures\\2019-12-03\\2019-12-03_193009.png";
//std::string image_path = "F:\\tmp\\new-libtorch\\lib-model-create\\timg1.jpg";
//int main(int argc, const char* argv[]) { 
//	/*if (argc != 2) {
//		std::cerr << "usage: example-app <path-to-exported-script-module>\n";
//		return -1;
//	}
//	*/
//	auto image = cv::imread(image_path);// , cv::IMREAD_COLOR/*cv::ImreadModes::IMREAD_IMREAD_COLOR*/);
//	cv::Mat image_transfomed;
//	cv::resize(image, image_transfomed, cv::Size(70, 70));
//	torch::jit::script::Module module;
//	torch::Tensor tensor_image = torch::from_blob(image_transfomed.data,
//	{ image_transfomed.rows, image_transfomed.cols,3 }, torch::kByte);
//	tensor_image = tensor_image.permute({ 2,0,1 });
//	tensor_image = tensor_image.toType(torch::kFloat);
//	tensor_image = tensor_image.div(255);
//	tensor_image = tensor_image.unsqueeze(0);
//	try {
//		// Deserialize the ScriptModule from a file using torch::jit::load(). 
//		//module = torch::jit::load("F:\\Temp File\\libtorch-win-shared-with-deps-131\\libtorch\\test\\model.pt");
//		module = torch::jit::load("F:\\tmp\\new-libtorch\\lib-model-create\\model_resnet_jit.pt");
//		//assert(module != nullptr);
//		//std::cout << "ok\n";
//		// Create a vector of inputs.
//		std::vector<torch::jit::IValue> inputs;
//		inputs.push_back(torch::ones({ 1, 3, 224, 224 }));
//
//		// Execute the model and turn its output into a tensor.
//		at::Tensor output = module.forward({ tensor_image }).toTensor();
//
//		auto prediction = output.argmax(1);
//		std::cout << "prediction:" << prediction << std::endl;
//
//		int maxk = 3;
//		auto top3 = std::get<1>(output.topk(maxk, 1, true, true));
//
//		std::cout << "top3: " << top3 << '\n';
//
//		std::vector<int> res;
//		for (auto i = 0; i < maxk; i++) {
//			res.push_back(top3[0][i].item().toInt());
//		}
//		for (auto i : res) {
//			std::cout << i << " ";
//		}
//		std::cout << "\n";
//
//		system("pause");
//
//		std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
//	}
//	catch (const c10::Error& e) {
//		std::cerr << "error loading the model\n";
//		return -1;
//	}
//	system("pause");
//	std::cout << "ok\n";
//}
/*
0.01 *
-9.8454 -66.8251 -20.5342 -65.0013 -31.0493
[ Variable[CPUFloatType]{1,5} ]
请按任意键继续. . .

0.01 *
-9.8454 -66.8251 -20.5342 -65.0013 -31.0493
[ Variable[CPUFloatType]{1,5} ]
请按任意键继续. . .

0.01 *
-2.9161 -84.5762 -15.0535 -80.4725 -25.2719
[ Variable[CPUFloatType]{1,5} ]
请按任意键继续. . .
*/
/*
#include <torch/script.h> // One-stop header.
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>

//https://pytorch.org/tutorials/advanced/cpp_export.html

std::string image_path = "your image folder path";

int main(int argc, const char* argv[]) {

// Deserialize the ScriptModule from a file using torch::jit::load().
std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("your model path");

assert(module != nullptr);
std::cout << "ok\n";

//输入图像
auto image = cv::imread(image_path +"/"+ "your image name",cv::ImreadModes::IMREAD_IMREAD_COLOR);
cv::Mat image_transfomed;
cv::resize(image, image_transfomed, cv::Size(70, 70));

// 转换为Tensor
torch::Tensor tensor_image = torch::from_blob(image_transfomed.data,
{image_transfomed.rows, image_transfomed.cols,3},torch::kByte);
tensor_image = tensor_image.permute({2,0,1});
tensor_image = tensor_image.toType(torch::kFloat);
tensor_image = tensor_image.div(255);
tensor_image = tensor_image.unsqueeze(0);

// 网络前向计算
at::Tensor output = module->forward({tensor_image}).toTensor();
//std::cout << "output:" << output << std::endl;

auto prediction = output.argmax(1);
std::cout << "prediction:" << prediction << std::endl;

int maxk = 3;
auto top3 = std::get<1>(output.topk(maxk, 1, true, true));

std::cout << "top3: " << top3 << '\n';

std::vector<int> res;
for (auto i = 0; i < maxk; i++) {
res.push_back(top3[0][i].item().toInt());
}
for (auto i : res) {
std::cout << i << " ";
}
std::cout << "\n";

system("pause");
}
*/