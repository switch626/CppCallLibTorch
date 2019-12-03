#include <torch/script.h> // One-stop header.
#include <opencv.hpp>
#include <iostream>
#include <memory>
std::string image_path = "H:\\项目\\BGSS\\sortsout\\Progress\\recordLog\\pictures\\2019-12-03\\2019-12-03_193009.png";
int main(int argc, const char* argv[]) {
	/*if (argc != 2) {
		std::cerr << "usage: example-app <path-to-exported-script-module>\n";
		return -1;
	}
	*/
	auto image = cv::imread(image_path, cv::IMREAD_COLOR/*cv::ImreadModes::IMREAD_IMREAD_COLOR*/);
	cv::Mat image_transfomed;
	cv::resize(image, image_transfomed, cv::Size(70, 70));
	torch::jit::script::Module module;
	torch::Tensor tensor_image = torch::from_blob(image_transfomed.data,
	{ image_transfomed.rows, image_transfomed.cols,3 }, torch::kByte);
	tensor_image = tensor_image.permute({ 2,0,1 });
	tensor_image = tensor_image.toType(torch::kFloat);
	tensor_image = tensor_image.div(255);
	tensor_image = tensor_image.unsqueeze(0);
	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		module = torch::jit::load("F:\\Temp File\\libtorch-win-shared-with-deps-131\\libtorch\\test\\model.pt");
		// Create a vector of inputs.
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(torch::ones({ 1, 3, 224, 224 }));

		// Execute the model and turn its output into a tensor.
		at::Tensor output = module.forward({ tensor_image }).toTensor();
		std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}
	system("pause");
	std::cout << "ok\n";
}
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