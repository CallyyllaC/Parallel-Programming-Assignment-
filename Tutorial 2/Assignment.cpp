#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <boost/compute.hpp>
#include "Utils.h"
#include "CImg.h"
using namespace cimg_library;
namespace compute = boost::compute; //not used yet (maybe in future)


//  ________                     ___________    ______   ________                                                         _____                
//  ___  __ \_____ _____________ ___  /__  /_______  /   ___  __ \___________________ _____________ _______ __________ ______(_)_____________ _
//  __  /_/ /  __ `/_  ___/  __ `/_  /__  /_  _ \_  /    __  /_/ /_  ___/  __ \_  __ `/_  ___/  __ `/_  __ `__ \_  __ `__ \_  /__  __ \_  __ `/
//  _  ____// /_/ /_  /   / /_/ /_  / _  / /  __/  /     _  ____/_  /   / /_/ /  /_/ /_  /   / /_/ /_  / / / / /  / / / / /  / _  / / /  /_/ / 
//  /_/     \__,_/ /_/    \__,_/ /_/  /_/  \___//_/      /_/     /_/    \____/_\__, / /_/    \__,_/ /_/ /_/ /_//_/ /_/ /_//_/  /_/ /_/_\__, /  
//                                                                            /____/                                                  /____/   

/*
	Assignment 1 Parallel Programming
	Callum Dyson-Gainsborough
	DYS17643101


	This code was tested and ran on an AMD GPU using OCL SDK Light, this code was also previously tested on the LAB pc’s using intel and Nvidia however
	due to being unable to access labs I cannot say for certain if the code will run effectively on these anymore.
	If the code is unable to be run, here is a video of the code running on my machine:
	https://youtu.be/vCGw29xD71o

	This implementation of using open cl to parallelise equalisation histograms uses multiple optimised parallel techniques including:
		Atomic addition
		Reduce addition
		Global memory
		Local memory
		HS scan
		Double buffered HS scan
		Blelloch scan

	At run time the program allows the user to change the following, each loop allowing different methods to be tested without restarting the program:
		Pixel bin sizes
		Work group sizes
		Open CL platform and device (will print out a full list of every platform and their respective devices for the user to pick from)
		Which image is to be used

	This program also supports both 8 and 16bit image inputs, in both ppm and pgm formats, as well as fully functioning with both grayscale and colour images.
	The colour imgae used is a jpg image that has been converted into 8bit ppm format.
	The 16bit image used is the same as the small 8bit image but exported with 16bit colour

	This outputs both equalised and normalised images using different techniques to show performance difference.

	The results for each open cl kernel is shown after its run in a table for the user, as well as after all kernels have run, all of the previous results
	are shown clearly in a table for quick and easy user comparison. This table includes: Queued, Submitted, Executed, and Total times.

*/


//           (_)           / _|                | | (_)                
//  _ __ ___  _ ___  ___  | |_ _   _ _ __   ___| |_ _  ___  _ __  ___
// | '_ ` _ \| / __|/ __| |  _| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
// | | | | | | \__ \ (__  | | | |_| | | | | (__| |_| | (_) | | | \__ \
// |_| |_| |_|_|___/\___| |_| \__, _|_| |_|\___|\__|_|\___/|_| |_|___/

//used to find the index values of the min and max values for normalisation
int FindMinIndex(const std::vector<int>& vec, int binSize)
{
	int kMin = 0;

	while (vec[kMin] == 0)
	{
		kMin += 1;
		if (kMin == binSize - 1)
		{
			return binSize - 1;
		}
	}
	return kMin;
}
int FindMaxIndex(const std::vector<int>& vec, int binSize)
{
	int kMax = binSize - 1;

	while (vec[kMax] == 0)
	{
		kMax -= 1;
		if (kMax == 0)
		{
			return 0;
		}
	}
	return kMax;
}

//used for ascii art
//fills out the table for standard spacing
string FillBox(string input, int size)
{
	int add = size - input.size();
	if (add < 0)
	{
		return "| " + input + " |";
	}

	for (size_t i = 0; i < add; i++)
	{
		if (i % 2 == 0 || i == 0)
		{
			input = " " + input;
		}
		else
		{
			input = input + " ";
		}
	}

	return "| " + input + " |";
}
//standardises spacing without adding table borders
string AddFill(char input, int size)
{
	string out = "";
	for (size_t i = 0; i < size; i++)
	{
		out = out + input;
	}

	return out;
}

//used to print the info of the kernal event
//this is printing in line during the main workflow and only provides the basic
void PrintEventInfo(cl::Event event)
{
	string out = "";
	string profiling = GetFullProfilingInfo(event, ProfilingResolution::PROF_NS);
	profiling = FillBox(GetFullProfilingInfo(event, ProfilingResolution::PROF_NS), profiling.size());
	string line = AddFill('-', profiling.size() + 34);
	string speed = FillBox(to_string(event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()), profiling.size() - 4);
	out = out + FillBox("Kernel timings:", 30) + profiling + "\n";
	out = out + line + "\n";
	out = out + FillBox("Kernel execution time (ns):", 30) + speed + "\n";

	std::cout << endl << endl << out << endl;
}

//used to print the info of the kernal event
//queued, submitted, executed and total timings presented in a table format
void PrintInfo(cl::Kernel ker, cl::Device device, int glob , int loc)
{
	string out = "";

	string pwgs = to_string(ker.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device));
	string gwg = to_string(glob);
	string lwg = to_string(loc);
	string line = AddFill('-', 88);

	pwgs = FillBox(pwgs, 50);
	gwg = FillBox(gwg, 50);
	lwg = FillBox(lwg, 50);

	out = out + FillBox("Preferred Work Group size:", 30) + pwgs + "\n";
	out = out + line + "\n";
	out = out + FillBox("Global Work Group size:", 30) + gwg + "\n";
	out = out + line + "\n";
	out = out + FillBox("Local Work Group size:", 30) + lwg + "\n";

	std::cout << endl << endl << out << endl;
}

//used to print the info of the kernal events
//prints a table at the end of the main workflow that presents all of the event timings for queued, submitted, executed, and total together to allow for easier comparison
void PrintSummary(std::vector <cl::Event> events, std::vector<string> names)
{
	std::cout << " ____ ____ ____ ____ ____ ____ ____ \n" <<
		"||s |||u |||m |||m |||a |||r |||y ||\n" <<
		"||__|||__|||__|||__|||__|||__|||__||\n" <<
		"|/__\\|/__\\|/__\\|/__\\|/__\\|/__\\|/__\\|\n" << endl << endl;

	int rows = events.size() + 1;
	int cols = 5;
	int size = 10;
	std::vector <string> colHeadings{ "Queued", "Submitted", "Executed", "Total" };
	string line;
	int len = 0;
	//for each row
	for (size_t i = 0; i < rows; i++)
	{
		string queued;
		string submitted;
		string executed;
		string total;
		if (i > 0)
		{
			//get the data
			std::size_t pos;
			string all = GetFullProfilingInfo(events[i - 1], ProfilingResolution::PROF_NS);
			pos = all.find("Queued");
			queued = all.substr(pos + 7, all.find(',') - (pos + 7));
			all = all.substr(all.find(',') + 1);
			pos = all.find("Submitted");
			submitted = all.substr(pos + 10, all.find(',') - (pos + 10));
			all = all.substr(all.find(',') + 1);
			pos = all.find("Executed");
			executed = all.substr(pos + 9, all.find(',') - (pos + 9));
			all = all.substr(all.find(',') + 1);
			pos = all.find("Total");
			total = all.substr(pos + 6, all.find('[') - (pos + 6) - 1);
		}

		//set the row string
		string row = "";

		//for each col
		for (size_t j = 0; j < cols; j++)
		{
			if (i == 0)
			{
				if (j != 0)
				{
					//write the column headings
					row = row + FillBox(colHeadings[j - 1], size);
				}
				else
				{
					row = "Events";
					row = FillBox(row, 50);
				}
			}
			else
			{
				switch (j)
				{
				case 0:
					//write the row heading
					row = row + FillBox(names[i - 1], 50);
					break;
				case 1:
					//write the data
					row = row + FillBox(queued, size);
					break;
				case 2:
					//write the data
					row = row + FillBox(submitted, size);
					break;
				case 3:
					//write the data
					row = row + FillBox(executed, size);
					break;
				case 4:
					//write the data
					row = row + FillBox(total, size);
					break;
				}
			}
		}

		//print the row
		if (i == 0)
		{
			len = row.size();
			line = AddFill('-', len);
			std::cout << row << endl;
			std::cout << line << endl;

		}
		else if (i == rows - 1)
		{
			std::cout << row << endl;
		}
		else
		{
			std::cout << row << endl;
			std::cout << line << endl;
		}
	}
}

//used to print the histogram
//generates an ascii art title for the histogram and then prints the histogram vector underneath
void PrintHistogram(std::vector <int> hist, string name)
{
	std::cout << endl << endl << endl;

	//print histogram name
	//each line
	for (size_t i = 0; i < 4; i++)
	{
		string out = " ";
		//each letter
		for (size_t j = 0; j < name.size(); j++)
		{
			switch (i)
			{
				case 0:
					out = out + " ____";
					break;
				case 1:
					out = out + "||" + name[j] + " |";
					break;
				case 2:
					out = out + "||__|";
					break;
				case 3:
					out = out + "|/__\\";
					break;
			}


			if (j == (name.size() - 1) && i != 0)
			{
				out = out + "|";
			}
		}
		std::cout << out << endl;
	}
	std::cout << endl << endl;

	//print histogram
	cout << hist << endl;

}

//used to generate and print the name if output is an image instead of a vector
void PrintHistogram(string name)
{
	std::cout << endl << endl << endl;

	//print histogram name
	for (size_t i = 0; i < 4; i++)
	{
		string out = " ";
		for (size_t j = 0; j < name.size(); j++)
		{
			switch (i)
			{
			case 0:
				out = out + " ____";
				break;
			case 1:
				out = out + "||" + name[j] + " |";
				break;
			case 2:
				out = out + "||__|";
				break;
			case 3:
				out = out + "|/__\\";
				break;
			}
			if (j == name.size() - 1 && i != 0)
			{
				out = out + "|";
			}
		}
		std::cout << out << endl;
	}
	std::cout << endl << endl;
}

//pad the vector out to be divisible by the current work group size
std::vector<int>Pad(vector<int> vec, int wgs)
{
	size_t size = vec.size() % wgs;

	//if not a multiple with the wg size, pad out with 0's and return
	if (size)
	{
		std::vector<int> vecExt(wgs - size, 0);
		vec.insert(vec.end(), vecExt.begin(), vecExt.end());
		return vecExt;
	}

	//or return as is
	return vec;
}

//select an image to use
string SelectImage() {
	bool Menu = true;
	int input;
	//while at the menu
	while (Menu)
	{

		//Write the main menu
		std::cout << "Please select a file to load:" << endl << "1. test.pgm" << endl << "2. test_large.pgm" << endl << "3. test_colour.ppm" << endl << "4. test_16.pgm" << endl << endl;

		//wait for user input and catch errors
		try
		{
			std::cin >> input;
		}
		catch (const std::exception & e)
		{
			std::cout << e.what() << endl;
		}
		std::cout << endl;

		switch (input)
		{
		case 1:
			return "test.pgm";
			break;
		case 2:
			return "test_large.pgm";
			break;
		case 3:
			return "test_colour.ppm";
			break;
		case 4:
			return "test_16.pgm";
			break;
		default:
			std::cout << endl << "Please select a valid option" << endl << endl;
			break;
		}

	}
}

//select a bin for the histograms
int SelectPixelBin() {
	bool Menu = true;
	int input;
	//while at the menu
	while (Menu)
	{

		//Write the main menu
		std::cout << "Please enter the required pixel bin size (def: 256):" << endl << endl;

		//wait for user input and catch errors
		try
		{
			std::cin >> input;
		}
		catch (const std::exception & e)
		{
			std::cout << e.what() << endl;
		}
		std::cout << endl;

		if (input > 1)
		{
			return input;
		}
		else
		{
			std::cout << endl << "Please select a valid integer" << endl << endl;
		}


	}
}

//select a bin for the kernals
int SelectBin() {
	bool Menu = true;
	int input;
	//while at the menu
	while (Menu)
	{

		//Write the main menu
		std::cout << "Please enter the required local work group size (def: 256):" << endl << endl;

		//wait for user input and catch errors
		try
		{
			std::cin >> input;
		}
		catch (const std::exception & e)
		{
			std::cout << e.what() << endl;
		}
		std::cout << endl;

		if (input > 0)
		{
			return input;
		}
		else
		{
			std::cout << endl << "Please select a integer greater than 0" << endl << endl;
		}


	}
}

//gets a list of all connected platforms then allow the user to select from a list of platforms
int GetPlatform() {
	//Select platform and device to use
	vector <cl::Platform> platforms;
	cl::Platform::get(&platforms);
	vector <cl::Device> platformDevices;
	cl::string platformName;
	int platform_id = 1000;

	std::cout << "Please select a platform:" << std::endl;
	for (size_t i = 0; i < platforms.size(); i++)
	{
		platforms[i].getInfo(CL_PLATFORM_NAME, &platformName);
		std::cout << i + 1 << ". " << platformName << std::endl;
	}
	while (platform_id == 1000)
	{
		std::cout << std::endl;
		try
		{
			int tmp;
			std::cin >> tmp;
			if (tmp <= platforms.size() && tmp > 0)
			{
				platform_id = tmp - 1;
				return platform_id;
			}
		}
		catch (const std::exception&)
		{
			std::cout << "Error Please try again" << std::endl;
		}
	}
}

//gets a list of all connected devices for the current platform then allow the user to select which device to use
int GetDevice(int id) {
	vector <cl::Platform> platforms;
	cl::Platform::get(&platforms);
	vector<cl::Device> devices;
	platforms[id].getDevices(CL_DEVICE_TYPE_ALL, &devices);
	int device_id = 1000;

	std::cout << "Please select a cl_device:" << std::endl;
	for (size_t i = 0; i < devices.size(); i++)
	{
		std::cout << i + 1 << ". " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
	}
	while (device_id == 1000)
	{
		std::cout << std::endl;
		try
		{
			int tmp;
			std::cin >> tmp;
			if (tmp <= devices.size() && tmp > 0)
			{
				device_id = tmp - 1;
				return device_id;
			}
		}
		catch (const std::exception&)
		{
			std::cout << "Error Please try again" << std::endl;
		}
	}
}

//display all of the detected platforms and their respective devices
void ShowPlatformsAndDevices() {
	vector <cl::Platform> platforms;
	vector <cl::Device> platformDevices;
	cl::Platform::get(&platforms);
	cl::string platformName;

	if (platforms.size() == 0)
	{
		std::cout << "No opencl platforms found" << std::endl;
	}
	else
	{
		std::cout << platforms.size() << " opencl platforms found" << std::endl;
	}

	for (size_t i = 0; i < platforms.size(); i++)
	{
		platformDevices.clear();

		platforms[i].getInfo(CL_PLATFORM_NAME, &platformName);
		std::cout << "Getting devices from " << platformName << std::endl;

		platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);


		if (platformDevices.size() == 0)
		{
			std::cout << "No opencl devices found for platform " << i << std::endl;
			break;
		}

		for (size_t j = 0; j < platformDevices.size(); j++)
		{
			std::cout << "Found cl_device: " << platformDevices[j].getInfo<CL_DEVICE_NAME>() << std::endl;
		}
		std::cout << std::endl;
	}
}


//                  _                            _     __ _               
//                 (_)                          | |   / _| |              
//  _ __ ___   __ _ _ _ __   __      _____  _ __| | _| |_| | _____      __
// | '_ ` _ \ / _` | | '_ \  \ \ /\ / / _ \| '__| |/ /  _| |/ _ \ \ /\ / /
// | | | | | | (_| | | | | |  \ V  V / (_) | |  |   <| | | | (_) \ V  V / 
// |_| |_| |_|\__,_|_|_| |_|   \_/\_/ \___/|_|  |_|\_\_| |_|\___/ \_/\_/  

int main(int argc, char** argv) {

	//main loop
	bool main = true;

	//configure errors
	cimg::exception_mode(0);

	//Write the main menu
	std::cout <<	" ____ ____ ____ ____ ____ ____ ____ ____ \n" <<
					"||P |||a |||r |||a |||l |||l |||e |||l ||\n" <<
					"||__|||__|||__|||__|||__|||__|||__|||__||\n" <<
					"|/__\\|/__\\|/__\\|/__\\|/__\\|/__\\|/__\\|/__\\|\n\n" <<
					" ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ \n" <<
					"||P |||r |||o |||g |||r |||a |||m |||m |||i |||n |||g ||\n" <<
					"||__|||__|||__|||__|||__|||__|||__|||__|||__|||__|||__||\n" <<
					"|/__\\|/__\\|/__\\|/__\\|/__\\|/__\\|/__\\|/__\\|/__\\|/__\\|/__\\|\n" << endl << endl;

	std::cout << "Parallel Programming: Assessment Item 1" << endl << "Callum Dyson-Gainsborough, 17643101" << endl << endl;

	//Main loop, keep restarting after finish
	while (main)
	{
		//Main Try-Catch
		try
		{

			//Select platform and device to use
			ShowPlatformsAndDevices();
			int cl_platformId = GetPlatform();
			int cl_deviceId = GetDevice(cl_platformId);

			//Setup Hardware and add kernals
			cl::Context cl_context = GetContext(cl_platformId, cl_deviceId);
			cl::CommandQueue cl_queue(cl_context, CL_QUEUE_PROFILING_ENABLE);
			cl::Program::Sources cl_sources;
			AddSources(cl_sources, "kernels/my_kernels.cl");
			cl::Program cl_program(cl_context, cl_sources);
			cl::Device cl_device = cl_context.getInfo<CL_CONTEXT_DEVICES>()[0];

			//build and debug the kernel code
			try
			{
				cl_program.build();
			}
			catch (const cl::Error & err)
			{
				std::cout << "Build Status: " << cl_program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(cl_context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
				std::cout << "Build Options:\t" << cl_program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(cl_context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
				std::cout << "Build Log:\t " << cl_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
				throw err;
			}

			//declare pixel bins and work group size
			int binSize = 256;
			int lwgs = 256;

			//select an image
			string imageFileName = SelectImage();

			//select pixel bin size
			binSize = SelectPixelBin();
			//Set work group size
			lwgs = SelectBin();


			//Main Variables
			std::cout << "Initialising Container Variables..." << endl;
			CImg<unsigned char> imageInput(imageFileName.c_str());	// holds the input image
			vector<int> intensityHistogram(binSize, 0);						// holds the intensity histogram
			vector<int> cumulativeHistogram(binSize, 0);			// holds the cumulative histogram
			vector<int> equalisedHistogram(binSize, 0);				// holds the equalised histogram
			vector<int> min(1, 0);									// holds intensity histogram min index
			vector<int> max(1, 0);									// holds intensity histogram max index
			vector<int> pixelBin(1, binSize);						// holds the pixel bin size
			vector<int> group(1, 0);								// holds the local bin size
			vector<unsigned char> equalisedOutputBuffer(imageInput.size());	// holds the output image (equalised)
			vector<unsigned char> NormalisedOutputBuffer(imageInput.size());	// holds the output image (normalised)

			//Events
			//Holds the events that are tied to kernal executions for debug and performance info
			std::cout << "Initialising Events..." << endl;
			cl::Event intensityHistogramEvent;
			cl::Event intensityHistogramEventLocal;
			cl::Event cumulativeHistogramEvent;
			cl::Event cumulativeHistogramEventLocal;
			cl::Event cumulativeHistogramEventBL;
			cl::Event equaliseHistogramEvent;
			cl::Event backProjectionEvent;
			cl::Event normaliseEvent;

			//Buffers
			//Used to pass variables to and from the kernals
			std::cout << "Initialising Buffers..." << endl;
			cl::Buffer imageInputBuffer(cl_context, CL_MEM_READ_ONLY, imageInput.size());				//set input image to read only
			cl::Buffer imageOutputBuffer(cl_context, CL_MEM_READ_WRITE, imageInput.size());			//allow other buffers to be wrote to
			cl::Buffer imageOutputBuffer2(cl_context, CL_MEM_READ_WRITE, imageInput.size());			//set sizes in runtime based on type and size
			cl::Buffer intensityHistogramBuffer(cl_context, CL_MEM_READ_WRITE, binSize * sizeof(int));
			cl::Buffer cumulativeHistogramBuffer(cl_context, CL_MEM_READ_WRITE, binSize * sizeof(int));
			cl::Buffer equalisedHistogramBuffer(cl_context, CL_MEM_READ_WRITE, binSize * sizeof(int));
			cl::Buffer minBuffer(cl_context, CL_MEM_READ_WRITE, sizeof(int));
			cl::Buffer maxBuffer(cl_context, CL_MEM_READ_WRITE, sizeof(int));
			cl::Buffer pixelBinBuffer(cl_context, CL_MEM_READ_WRITE, sizeof(int));
			cl::Buffer groupBuffer(cl_context, CL_MEM_READ_WRITE, sizeof(int));

			//fill buffers with 0's
			//queue.enqueueFillBuffer(imageInputBuffer);


			//Pad the vector to run more efficiently on specific work group sizes
			intensityHistogram = Pad(intensityHistogram, lwgs);
			cumulativeHistogram = Pad(cumulativeHistogram, lwgs);
			equalisedHistogram = Pad(equalisedHistogram, lwgs);

			//Kernals
			//Used the inferface with OpenCL
			std::cout << "Initialising Kernals..." << endl;

			//Setup the intensity histogram kernal
			cl::Kernel ker_intensityHistogram = cl::Kernel(cl_program, "intensityHistogram");
			ker_intensityHistogram.setArg(0, imageInputBuffer);						// A = Input image buffer
			ker_intensityHistogram.setArg(1, intensityHistogramBuffer);				// B = Intensity histogram buffer
			ker_intensityHistogram.setArg(2, pixelBinBuffer);						// C = Pixel bin size

			//Setup the local intensity histogram kernal
			cl::Kernel ker_intensityHistogramLocal = cl::Kernel(cl_program, "intensityHistogramLocal");
			ker_intensityHistogramLocal.setArg(0, imageInputBuffer);					// A = Input image buffer
			ker_intensityHistogramLocal.setArg(1, intensityHistogramBuffer);			// B = Intensity histogram buffer
			ker_intensityHistogramLocal.setArg(2, pixelBinBuffer);						// C = Bin size buffer
			ker_intensityHistogramLocal.setArg(3, cl::Local(lwgs * sizeof(int)));		// D = Local buffer
			ker_intensityHistogramLocal.setArg(4, groupBuffer);							// E = Local work group size 

			//Setup the hs histogram kernal
			cl::Kernel ker_cumulativeHistogram = cl::Kernel(cl_program, "cumulativeHistogram");
			ker_cumulativeHistogram.setArg(0, intensityHistogramBuffer);			// A = Intensity histogram buffer
			ker_cumulativeHistogram.setArg(1, cumulativeHistogramBuffer);			// B = Cumulitive histogram buffer

			//Setup the db hs histogram kernal
			cl::Kernel ker_cumulativeHistogramLocal = cl::Kernel(cl_program, "cumulativeHistogramLocal");
			ker_cumulativeHistogramLocal.setArg(0, intensityHistogramBuffer);			// A = Input image buffer
			ker_cumulativeHistogramLocal.setArg(1, cumulativeHistogramBuffer);			// B = Intensity histogram buffer
			ker_cumulativeHistogramLocal.setArg(2, cl::Local(lwgs * sizeof(int)));		// C = Local buffer
			ker_cumulativeHistogramLocal.setArg(3, cl::Local(lwgs * sizeof(int)));		// D = Local buffer

			//Setup the bl histogram kernal
			cl::Kernel ker_cumulativeHistogramBL = cl::Kernel(cl_program, "cumulativeHistogram_bl");
			ker_cumulativeHistogramBL.setArg(0, intensityHistogramBuffer);			// A = Input image buffer
			ker_cumulativeHistogramBL.setArg(1, cumulativeHistogramBuffer);			// B = Intensity histogram buffer

			//Setup the equalisation histogram kernal
			cl::Kernel ker_equalisedHistogram = cl::Kernel(cl_program, "equalisedHistogram");
			ker_equalisedHistogram.setArg(0, cumulativeHistogramBuffer);			// A = Cumulitive histogram buffer
			ker_equalisedHistogram.setArg(1, equalisedHistogramBuffer);				// B = Equalised histogram buffer

			//Setup the LUT kernal
			cl::Kernel ker_backProjection = cl::Kernel(cl_program, "backProjection");
			ker_backProjection.setArg(0, equalisedHistogramBuffer);					// A = Equalised histogram buffer
			ker_backProjection.setArg(1, imageInputBuffer);							// B = Input image buffer
			ker_backProjection.setArg(2, imageOutputBuffer);						// C = Output image buffer
			ker_backProjection.setArg(3, pixelBinBuffer);							// D = Bin size buffer

			//Setup the normalisation kernal
			cl::Kernel ker_normaliseImage = cl::Kernel(cl_program, "normaliseImage");
			ker_normaliseImage.setArg(0, imageInputBuffer);					// A = Input image buffer
			ker_normaliseImage.setArg(1, imageOutputBuffer2);				// B = Output image buffer
			ker_normaliseImage.setArg(2, minBuffer);						// C = Intensity histogram min index buffer
			ker_normaliseImage.setArg(3, maxBuffer);						// D = Intensity histogram max index buffer

			// Variables and kernals have all been declared
			// now to move on to the Core Workflow and call the kernals
			// take the outputs of these kernals and then store them ready to be used by the next kernal

			std::cout << endl << "Starting..." << endl;

			//  _                   _     _                 
			// (_)                 | |   (_)                
			//  _ _ __  _ __  _   _| |_   _ _ __ ___   __ _ 
			// | | '_ \| '_ \| | | | __| | | '_ ` _ \ / _` |
			// | | | | | |_) | |_| | |_  | | | | | | | (_| |
			// |_|_| |_| .__/ \__,_|\__| |_|_| |_| |_|\__, |
			//         | |                             __/ |
			//         |_|                            |___/

			std::cout << endl << "Working on input image..." << endl;
			CImgDisplay disp_input(imageInput, "Input Image");

			//  _       _                 _ _           _     _     _   
			// (_)     | |               (_) |         | |   (_)   | |  
			//  _ _ __ | |_ ___ _ __  ___ _| |_ _   _  | |__  _ ___| |_ 
			// | | '_ \| __/ _ \ '_ \/ __| | __| | | | | '_ \| / __| __|
			// | | | | | ||  __/ | | \__ \ | |_| |_| | | | | | \__ \ |_ 
			// |_|_| |_|\__\___|_| |_|___/_|\__|\__, | |_| |_|_|___/\__|
			//                                   __/ |                  
			//                                  |___/

			//Intensity Histogram Method 1

			//Write the vectors into the buffers
			cl_queue.enqueueWriteBuffer(imageInputBuffer, CL_TRUE, 0, imageInput.size(), &imageInput.data()[0]);
			cl_queue.enqueueWriteBuffer(intensityHistogramBuffer, CL_TRUE, 0, binSize * sizeof(int), &intensityHistogram[0]);
			cl_queue.enqueueWriteBuffer(pixelBinBuffer, CL_TRUE, 0, sizeof(int), &pixelBin[0]);

			//Queue the kernal with the correct wgs and event
			cl_queue.enqueueNDRangeKernel(ker_intensityHistogram, cl::NullRange, cl::NDRange(imageInput.size()), cl::NDRange(lwgs), NULL, &intensityHistogramEvent);

			//Read the output back from the buffer and store in the vector
			cl_queue.enqueueReadBuffer(intensityHistogramBuffer, CL_TRUE, 0, binSize * sizeof(int), &intensityHistogram[0]);

			//Print Details and Results
			PrintHistogram(intensityHistogram, "Intensity Histogram");
			PrintInfo(ker_intensityHistogram, cl_device, imageInput.size(), lwgs);
			PrintEventInfo(intensityHistogramEvent);

			//Intensity Histogram Method 2

			//Write the vectors into the buffers
			cl_queue.enqueueWriteBuffer(imageInputBuffer, CL_TRUE, 0, imageInput.size(), &imageInput.data()[0]);
			cl_queue.enqueueWriteBuffer(intensityHistogramBuffer, CL_TRUE, 0, binSize * sizeof(int), &intensityHistogram[0]);
			cl_queue.enqueueWriteBuffer(pixelBinBuffer, CL_TRUE, 0, sizeof(int), &pixelBin[0]);
			cl_queue.enqueueWriteBuffer(groupBuffer, CL_TRUE, 0, sizeof(int), &group[0]);

			//Queue the kernal with the correct wgs and event
			cl_queue.enqueueNDRangeKernel(ker_intensityHistogramLocal, cl::NullRange, cl::NDRange(imageInput.size()), cl::NDRange(lwgs), NULL, &intensityHistogramEventLocal);

			//Read the output back from the buffer and store in the vector
			cl_queue.enqueueReadBuffer(intensityHistogramBuffer, CL_TRUE, 0, binSize * sizeof(int), &intensityHistogram[0]);

			//Print Details and Results
			PrintHistogram(intensityHistogram, "Local Intensity Histogram");
			PrintInfo(ker_intensityHistogramLocal, cl_device, imageInput.size(), lwgs);
			PrintEventInfo(intensityHistogramEventLocal);

			//                             _       _   _             _     _     _   
			//                            | |     | | (_)           | |   (_)   | |  
			//   ___ _   _ _ __ ___  _   _| | __ _| |_ ___   _____  | |__  _ ___| |_ 
			//  / __| | | | '_ ` _ \| | | | |/ _` | __| \ \ / / _ \ | '_ \| / __| __|
			// | (__| |_| | | | | | | |_| | | (_| | |_| |\ V /  __/ | | | | \__ \ |_ 
			//  \___|\__,_|_| |_| |_|\__,_|_|\__,_|\__|_| \_/ \___| |_| |_|_|___/\__|

			//Cumulative Histogram Method 1

			//Write the vectors into the buffers
			cl_queue.enqueueWriteBuffer(intensityHistogramBuffer, CL_TRUE, 0, binSize * sizeof(int), &intensityHistogram[0]);
			cl_queue.enqueueWriteBuffer(cumulativeHistogramBuffer, CL_TRUE, 0, binSize * sizeof(int), &cumulativeHistogram[0]);

			//Queue the kernal with the correct wgs and event
			cl_queue.enqueueNDRangeKernel(ker_cumulativeHistogramBL, cl::NullRange, cl::NDRange(binSize), cl::NDRange(lwgs), NULL, &cumulativeHistogramEventBL);

			//Read the output back from the buffer and store in the vector
			cl_queue.enqueueReadBuffer(cumulativeHistogramBuffer, CL_TRUE, 0, binSize * sizeof(int), &cumulativeHistogram[0]);

			//Print Details and Results
			PrintHistogram(cumulativeHistogram, "BL Cumulative Histogram");
			PrintInfo(ker_cumulativeHistogramBL, cl_device, binSize, lwgs);
			PrintEventInfo(cumulativeHistogramEventBL);

			//Cumulative Histogram Method 2

			//Write the vectors into the buffers
			cl_queue.enqueueWriteBuffer(intensityHistogramBuffer, CL_TRUE, 0, binSize * sizeof(int), &intensityHistogram[0]);
			cl_queue.enqueueWriteBuffer(cumulativeHistogramBuffer, CL_TRUE, 0, binSize * sizeof(int), &cumulativeHistogram[0]);

			//Queue the kernal with the correct wgs and event
			cl_queue.enqueueNDRangeKernel(ker_cumulativeHistogram, cl::NullRange, cl::NDRange(binSize), cl::NDRange(lwgs), NULL, &cumulativeHistogramEvent);

			//Read the output back from the buffer and store in the vector
			cl_queue.enqueueReadBuffer(cumulativeHistogramBuffer, CL_TRUE, 0, binSize * sizeof(int), &cumulativeHistogram[0]);

			//Fix bug counting down
			for (size_t i = 1; i < binSize; i++)
			{
				if (cumulativeHistogram[i] < cumulativeHistogram[i-1])
				{
					cumulativeHistogram[i] = cumulativeHistogram[i - 1];
				}
			}

			//Print Details and Results
			PrintHistogram(cumulativeHistogram, "HS Cumulative Histogram");
			PrintInfo(ker_cumulativeHistogram, cl_device, binSize, lwgs);
			PrintEventInfo(cumulativeHistogramEvent);

			//Cumulative Histogram Method 3

			//Write the vectors into the buffers
			cl_queue.enqueueWriteBuffer(intensityHistogramBuffer, CL_TRUE, 0, binSize * sizeof(int), &intensityHistogram[0]);
			cl_queue.enqueueWriteBuffer(cumulativeHistogramBuffer, CL_TRUE, 0, binSize * sizeof(int), &cumulativeHistogram[0]);

			//Queue the kernal with the correct wgs and event
			cl_queue.enqueueNDRangeKernel(ker_cumulativeHistogramLocal, cl::NullRange, cl::NDRange(binSize), cl::NDRange(lwgs), NULL, &cumulativeHistogramEventLocal);

			//Read the output back from the buffer and store in the vector
			cl_queue.enqueueReadBuffer(cumulativeHistogramBuffer, CL_TRUE, 0, binSize * sizeof(int), &cumulativeHistogram[0]);

			//Print Details and Results
			PrintHistogram(cumulativeHistogram, "DB HS Cumulative Histogram");
			PrintInfo(ker_cumulativeHistogramLocal, cl_device, binSize, lwgs);
			PrintEventInfo(cumulativeHistogramEventLocal);

			//                         _ _              _   _     _     _   
			//                        | (_)            | | | |   (_)   | |  
			//   ___  __ _ _   _  __ _| |_ ___  ___  __| | | |__  _ ___| |_ 
			//  / _ \/ _` | | | |/ _` | | / __|/ _ \/ _` | | '_ \| / __| __|
			// |  __/ (_| | |_| | (_| | | \__ \  __/ (_| | | | | | \__ \ |_ 
			//  \___|\__, |\__,_|\__,_|_|_|___/\___|\__,_| |_| |_|_|___/\__|
			//          | |                                                 
			//          |_|

			//Write the vectors into the buffers
			cl_queue.enqueueWriteBuffer(cumulativeHistogramBuffer, CL_TRUE, 0, binSize * sizeof(int), &cumulativeHistogram[0]);

			//Queue the kernal with the correct wgs and event
			cl_queue.enqueueNDRangeKernel(ker_equalisedHistogram, cl::NullRange, cl::NDRange(binSize), cl::NDRange(lwgs), NULL, &equaliseHistogramEvent);

			//Read the output back from the buffer and store in the vector
			cl_queue.enqueueReadBuffer(equalisedHistogramBuffer, CL_TRUE, 0, binSize * sizeof(int), &equalisedHistogram[0]);

			//Print Details and Results
			PrintHistogram(equalisedHistogram, "Equalised Histogram");
			PrintInfo(ker_equalisedHistogram, cl_device, binSize, lwgs);
			PrintEventInfo(equaliseHistogramEvent);


			if (binSize == 256) //normalise only supports 256 bins currently
			{	
				//                                   _ _              _   _     _     _   
				//                                  | (_)            | | | |   (_)   | |  
				//  _ __   ___  _ __ _ __ ___   __ _| |_ ___  ___  __| | | |__  _ ___| |_ 
				// | '_ \ / _ \| '__| '_ ` _ \ / _` | | / __|/ _ \/ _` | | '_ \| / __| __|
				// | | | | (_) | |  | | | | | | (_| | | \__ \  __/ (_| | | | | | \__ \ |_ 
				// |_| |_|\___/|_|  |_| |_| |_|\__,_|_|_|___/\___|\__,_| |_| |_|_|___/\__|

				//find id of first value in array that isnt 0
				min[0] = FindMinIndex(intensityHistogram, binSize);
				//find id of last value in array that isnt 0
				max[0] = FindMaxIndex(intensityHistogram, binSize);

				//Write the vectors into the buffers
				cl_queue.enqueueWriteBuffer(minBuffer, CL_TRUE, 0, sizeof(int), &min[0]);
				cl_queue.enqueueWriteBuffer(maxBuffer, CL_TRUE, 0, sizeof(int), &max[0]);

				//Queue the kernal with the correct wgs and event
				cl_queue.enqueueNDRangeKernel(ker_normaliseImage, cl::NullRange, cl::NDRange(imageInput.size()), cl::NDRange(lwgs), NULL, &normaliseEvent);

				//Read the output back from the buffer and store in the vector
				cl_queue.enqueueReadBuffer(imageOutputBuffer2, CL_TRUE, 0, NormalisedOutputBuffer.size(), &NormalisedOutputBuffer.data()[0]);

				//Print Details and Results
				PrintHistogram("Normalised Histogram");
				PrintInfo(ker_normaliseImage, cl_device, imageInput.size(), lwgs);
				PrintEventInfo(normaliseEvent);
			}

			//  _                _                      _           _   _             
			// | |              | |                    (_)         | | (_)            
			// | |__   __ _  ___| | __  _ __  _ __ ___  _  ___  ___| |_ _  ___  _ __  
			// | '_ \ / _` |/ __| |/ / | '_ \| '__/ _ \| |/ _ \/ __| __| |/ _ \| '_ \ 
			// | |_) | (_| | (__|   <  | |_) | | | (_) | |  __/ (__| |_| | (_) | | | |
			// |_.__/ \__,_|\___|_|\_\ | .__/|_|  \___/| |\___|\___|\__|_|\___/|_| |_|
			//                         | |            _/ |                            
			//                         |_|           |__/  

			//Write the vectors into the buffers
			cl_queue.enqueueWriteBuffer(equalisedHistogramBuffer, CL_TRUE, 0, binSize * sizeof(int), &equalisedHistogram[0]);

			//Queue the kernal with the correct wgs and event
			cl_queue.enqueueNDRangeKernel(ker_backProjection, cl::NullRange, cl::NDRange(imageInput.size()), cl::NDRange(lwgs), NULL, &backProjectionEvent);;

			//Read the output back from the buffer and store in the vector
			cl_queue.enqueueReadBuffer(imageOutputBuffer, CL_TRUE, 0, equalisedOutputBuffer.size(), &equalisedOutputBuffer.data()[0]);

			//Print Details and Results
			PrintHistogram("Back Projection");
			PrintInfo(ker_backProjection, cl_device, imageInput.size(), lwgs);
			PrintEventInfo(backProjectionEvent);

			//             _       _     _        __      
			//            (_)     | |   (_)      / _|     
			//  _ __  _ __ _ _ __ | |_   _ _ __ | |_ ___  
			// | '_ \| '__| | '_ \| __| | | '_ \|  _/ _ \ 
			// | |_) | |  | | | | | |_  | | | | | || (_) |
			// | .__/|_|  |_|_| |_|\__| |_|_| |_|_| \___/ 
			// | |                                        
			// |_|

			//declare a vector with all of the events and with all of the names of these events because aparently dictionaries dont work in c++
			std::vector<cl::Event> Summary;
			std::vector<string> SummaryNames;

			//include normalise if pixel bin is 256
			if (binSize != 256)
			{
				Summary = { intensityHistogramEvent, intensityHistogramEventLocal, cumulativeHistogramEvent, cumulativeHistogramEventLocal, cumulativeHistogramEventBL, equaliseHistogramEvent , backProjectionEvent };
				SummaryNames = { "Intensity Histogram","Intensity Histogram Local", "Hillis-Steele Scan", "Double Buffered Hillis-Steele Scan", "Blelloch Scan", "Equalised Histogram" , "Back Projection" };
			}
			else
			{
				Summary = { intensityHistogramEvent, intensityHistogramEventLocal, cumulativeHistogramEvent, cumulativeHistogramEventLocal, cumulativeHistogramEventBL, equaliseHistogramEvent, normaliseEvent, backProjectionEvent };
				SummaryNames = { "Intensity Histogram","Intensity Histogram Local", "Hillis-Steele Scan", "Double Buffered Hillis-Steele Scan", "Blelloch Scan", "Equalised Histogram" , "Normalisation", "Back Projection" };
			}
			
			//print to console
			PrintSummary(Summary, SummaryNames);

			//      _                     _                                 
			//     | |                   (_)                                
			//  ___| |__   _____      __  _ _ __ ___   __ _  ___
			// / __| '_ \ / _ \ \ /\ / / | | '_ ` _ \ / _` |/ __|
			// \__ \ | | | (_) \ V  V /  | | | | | | | (_| |\__ \
			// |___/_| |_|\___/ \_/\_/   |_|_| |_| |_|\__, ||___/
			//                                         __/ |          
			//                                        |___/        

			//write image vectors to CImg objs
			CImg<unsigned char> output_image(equalisedOutputBuffer.data(), imageInput.width(), imageInput.height(), imageInput.depth(), imageInput.spectrum());
			CImgDisplay disp_output(output_image, "Output Image Equalised");
			CImg<unsigned char> output_image2;
			CImgDisplay disp_output2;

			//write normalise image vector to CImg obj if 256 pixel bins
			if (binSize == 256)
			{
				output_image2 = CImg<unsigned char>(NormalisedOutputBuffer.data(), imageInput.width(), imageInput.height(), imageInput.depth(), imageInput.spectrum());
				disp_output2 = CImgDisplay(output_image2, "Output Image Normalised");
			}

			std::cout << endl << "Program finished, please close image windows to restart!" << endl;

			//wait until input or output image is closed
			while (!disp_input.is_closed() && !disp_input.is_keyESC() && !disp_output.is_closed() && !disp_output.is_keyESC())
			{
				disp_input.wait(1);
				disp_output.wait(1);
			}

			//loop back to the main menu
		}
		//Error handling
		catch (const cl::Error & err) {
			std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
		}
		catch (CImgException & err) {
			std::cerr << "ERROR: " << err.what() << endl;
		}

	}
	return 0;
}
