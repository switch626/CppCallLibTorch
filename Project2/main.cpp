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
	//����һ��cpu�豸���󣬺�������
	torch::Device cpu_device(torch::kCPU);

	//detect GPU
	//��ѯgpu��������auto_device����������gpu/cpu�Զ�ѡ��
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
	//�����л�pytorchģ�ͣ������ص�module����
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
	//�����inputs�������Ǹ��̶����͵ı���
	std::vector<torch::jit::IValue> inputs;

	//try to read a image to run the model
	//make the sequence in batch,channel,w,h
	//�˴�����Ҫʹ��opencv��ȡ����ͼƬ
	std::vector<cv::Mat> image_vec; //������ǻ�Ѷ�ȡ��������ͼƬ�Ž����vector�У������������
	std::vector<cv::Mat>& r_image_vec = image_vec;//����vector�����ã����㵱����������ʹ��
	std::vector<torch::Tensor> img_t;//����Ҫ��opencv��ȡ��matת����libtorch��tensor���������ﴴ��vector���������ʹ��
	std::vector<torch::Tensor>& r_img_t = img_t;//tensor vector�����ã����㵱����������ʹ��

	//��ȡͼƬ��������image_vec�У��������ʹ��
	readMatImg(argv[2], r_image_vec);
	//��image_vec����ȡͼƬ����ת����tensor��ʽ�������img_t��vector�У��Ա���沽��ʹ��
	prepareImgTensor(r_img_t, r_image_vec, 0);

	//�ظ����沽�裬��ȡ�ڶ���ͼƬ����Ϊ���Ǻ���Ҫ��ʾbatch��input������ʹ��������ͼƬ�������ֻ��һ��ͼƬ��û��ϵ��vector�з���һ��tensor������
	readMatImg(argv[3], r_image_vec);
	prepareImgTensor(r_img_t, r_image_vec, 1);

	//move to device
	//��tensor���뵽device�У�����auto_device�����ϵͳ��ʵ�������ѡ���Ƿ����gpu
	img_t[0] = img_t[0].to(auto_device);
	img_t[1] = img_t[1].to(auto_device);

	//������inference������Ҫ��tensor��autograd�ص�����Ȼ�˷��Դ���Դ
	auto img_var = torch::autograd::make_variable(img_t[0], false);
	auto img_var1 = torch::autograd::make_variable(img_t[1], false);

	//���������ᵽ��������ʹ��������ͼ�������أ�����Ҫ������tensorƴ�ӳ�������ʽ
	// [[1,3,224,224],
	//  [1,3,224,224]]
	//����ϵͳ���ܲ��д�������ͼƬ��
	auto img_vec = torch::cat({ img_var,img_var1 }, 0);
	//cat���tensor��Ҫpush��inputs�ı�����
	inputs.push_back(img_vec);
	std::cout << "inputs:" << inputs.size() << std::endl;

	std::cout << "start..." << std::endl;
	//��ʼ����ģ�ͽ���inference������ȡ���out_tensor
	torch::Tensor out_tensor = module.forward(inputs).toTensor();
	//std::cout<<"end..."<<"out tensor size:"<<out_tensor.sizes()<<std::endl;
	//���ڴ�ǰ������pytorch����ģ�͵�ʱ�����һ��ģ�͵������log_softmax��������������Ҫתһ�£����softmax
	out_tensor = torch::softmax(out_tensor, 1);
	//�ǵðѽ����ȡ��cpu���������������
	out_tensor = out_tensor.to(cpu_device);
	//��������һ��out_tensor�����ã������洦��������ʹ��
	torch::Tensor& r_out_tensor = out_tensor;
	//����Ҫ��out_tensor����ȡ������Ҫ�ķ���index��confidence,������ο�getInferenceResults�ĺ���˵��
	std::vector<std::tuple<int, float>> ret = getInferenceResults(r_out_tensor);

	for (auto& r : ret)
	{
		std::cout << "id:" << std::get<0>(r) << " conf:" << std::get<1>(r) << std::endl;
	}

	return 0;

}

//std::string image_path = "H:\\��Ŀ\\BGSS\\sortsout\\Progress\\recordLog\\pictures\\2019-12-03\\2019-12-03_193009.png";
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
�밴���������. . .

0.01 *
-9.8454 -66.8251 -20.5342 -65.0013 -31.0493
[ Variable[CPUFloatType]{1,5} ]
�밴���������. . .

0.01 *
-2.9161 -84.5762 -15.0535 -80.4725 -25.2719
[ Variable[CPUFloatType]{1,5} ]
�밴���������. . .
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

//����ͼ��
auto image = cv::imread(image_path +"/"+ "your image name",cv::ImreadModes::IMREAD_IMREAD_COLOR);
cv::Mat image_transfomed;
cv::resize(image, image_transfomed, cv::Size(70, 70));

// ת��ΪTensor
torch::Tensor tensor_image = torch::from_blob(image_transfomed.data,
{image_transfomed.rows, image_transfomed.cols,3},torch::kByte);
tensor_image = tensor_image.permute({2,0,1});
tensor_image = tensor_image.toType(torch::kFloat);
tensor_image = tensor_image.div(255);
tensor_image = tensor_image.unsqueeze(0);

// ����ǰ�����
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