// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"

// Create a macro to quickly mark a function for export
#define DLLExport __declspec (dllexport)

// Wrap code to prevent name-mangling issues
extern "C" {

	// The current source image width
	int img_w;
	// The current source image height
	int img_h;
	// The current model input width
	int input_w;
	// The current model input height
	int input_h;
	// The total number pixels in the input image
	int n_pixels;
	// The number of color channels 
	int n_channels = 3;

	// Stores information about a single object prediction
	struct Object
	{
		float x0;
		float y0;
		float width;
		float height;
		int label;
		float prob;
	};

	// Store grid offset and stride values to decode a section of the model output
	struct GridAndStride
	{
		int grid0;
		int grid1;
		int stride;
	};

	// The scale values used to adjust the model output to the source image resolution
	float scale_x;
	float scale_y;

	// The minimum confidence score to consider an object proposal
	float bbox_conf_thresh = 0.3;
	// The maximum intersection over union value before an object proposal will be ignored
	float nms_thresh = 0.45;

	// Stores the grid and stride values to navigate the raw model output
	std::vector<GridAndStride> grid_strides;
	// Stores the object proposals with confidence scores above bbox_conf_thresh
	std::vector<Object> proposals;
	// Stores the indices for the object proposals selected using non-maximum suppression
	std::vector<int> proposal_indices;

	// The stride values used to generate the gride_strides vector
	std::vector<int> strides = { 8, 16, 32 };

	// The mean of the ImageNet dataset used to train the model
	const float mean[] = { 0.485, 0.456, 0.406 };
	// The standard deviation of the ImageNet dataset used to train the model
	const float std_dev[] = { 0.229, 0.224, 0.225 };

	// ONNX Runtime API interface
	const OrtApi* ort = NULL;

	// List of available execution providers
	char** provider_names;
	int provider_count;

	// Holds the logging state for the ONNX Runtime objects
	OrtEnv* env;
	// Holds the options used when creating a new ONNX Runtime session
	OrtSessionOptions* session_options;
	// The ONNX Runtime session
	OrtSession* session;

	// The name of the model input
	char* input_name;
	// The name of the model output
	char* output_name;

	// A pointer to the raw input data
	float* input_data;
	// The memory size of the raw input data
	int input_size;


	/// <summary>
	/// Convert char data to wchar_t
	/// </summary>
	/// <param name="text"></param>
	/// <returns></returns>
	static wchar_t* charToWChar(const char* text)
	{
		const size_t size = strlen(text) + 1;
		wchar_t* wText = new wchar_t[size];
		size_t converted_chars;
		mbstowcs_s(&converted_chars, wText, size, text, _TRUNCATE);
		return wText;
	}

	/// <summary>
	/// Initialize the ONNX Runtime API interface and get the available execution providers
	/// </summary>
	/// <returns></returns>
	DLLExport void InitOrtAPI() {

		ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

		ort->GetAvailableProviders(&provider_names, &provider_count);
	}

	/// <summary>
	/// Get the number of available execution providers
	/// </summary>
	/// <returns>The number of available devices</returns>
	DLLExport int GetProviderCount()
	{
		// Return the number of available execution providers
		return provider_count;
	}

	/// <summary>
	/// Get the name of the execution provider at the specified index
	/// </summary>
	/// <param name="index"></param>
	/// <returns>The name of the execution provider at the specified index</returns>
	DLLExport char* GetProviderName(int index) {
		return provider_names[index];
	}

	/// <summary>
	/// Generate offset values to navigate the raw model output
	/// </summary>
	/// <param name="height">The model input height</param>
	/// <param name="width">The model input width</param>
	void GenerateGridsAndStride(int height, int width)
	{
		// Remove the values for the previous input resolution
		grid_strides.clear();

		// Iterate through each stride value
		for (auto stride : strides)
		{
			// Calculate the grid dimensions
			int grid_height = height / stride;
			int grid_width = width / stride;

			// Store each combination of grid coordinates
			for (int g1 = 0; g1 < grid_height; g1++)
			{
				for (int g0 = 0; g0 < grid_width; g0++)
				{
					grid_strides.push_back(GridAndStride{ g0, g1, stride });
				}
			}
		}
	}

	/// <summary>
	/// Set minimum confidence score for keeping bounding box proposals
	/// </summary>
	/// <param name="min_confidence">The minimum confidence score for keeping bounding box proposals</param>
	DLLExport void SetConfidenceThreshold(float min_confidence)
	{
		bbox_conf_thresh = min_confidence;
	}

	/// <summary>
	/// Refresh memory when switching models or execution providers
	/// </summary>
	DLLExport void RefreshMemory() {
		if (input_data) free(input_data);
		if (session) ort->ReleaseSession(session);
		if (env) ort->ReleaseEnv(env);
	}

	/// <summary>
	/// Load a model from the specified file path
	/// </summary>
	/// <param name="model_path">The full model path to the ONNX model</param>
	/// <param name="execution_provider">The name for the desired execution_provider</param>
	/// <param name="image_dims">The source image dimensions</param>
	/// <returns>A status value indicating success or failure to load and reshape the model</returns>
	DLLExport int LoadModel(char* model_path, char* execution_provider, int image_dims[2])
	{
		int return_val = 0;

		// Initialize the ONNX runtime environment
		std::string instance_name = "yolox-inference";
		ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, instance_name.c_str(), &env);

		// Disable telemetry
		ort->DisableTelemetryEvents(env);

		// Add the selected execution provider
		ort->CreateSessionOptions(&session_options);
		std::string provider_name = execution_provider;

		if (provider_name.find("CPU") != std::string::npos) {
			return_val = 1;
		}
		else if (provider_name.find("Dml") != std::string::npos) {
			ort->DisableMemPattern(session_options);
			ort->SetSessionExecutionMode(session_options, ExecutionMode::ORT_SEQUENTIAL);
			OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0);
		}
		else return_val = 1;

		// Create a new inference session
		ort->CreateSession(env, charToWChar(model_path), session_options, &session);
		ort->ReleaseSessionOptions(session_options);

		Ort::AllocatorWithDefaultOptions allocator;

		// Get input and output names
		ort->SessionGetInputName(session, 0, allocator, &input_name);
		ort->SessionGetOutputName(session, 0, allocator, &output_name);

		// The dimensions of the source input image
		img_w = image_dims[0];
		img_h = image_dims[1];
		// Calculate new input dimensions based on the max stride value
		input_w = (int)(strides.back() * std::roundf(img_w / strides.back()));
		input_h = (int)(strides.back() * std::roundf(img_h / strides.back()));
		n_pixels = input_w * input_h;

		// Calculate the value used to adjust the model output to the source image resolution
		scale_x = input_w / (img_w * 1.0);
		scale_y = input_h / (img_h * 1.0);

		// Generate the grid and stride values based on input resolution
		GenerateGridsAndStride(input_h, input_w);

		// Replace the initial input dims with the updated values
		image_dims[0] = input_w;
		image_dims[1] = input_h;

		// Allocate memory for the raw input data
		input_size = n_pixels * n_channels * (int)sizeof(float);
		input_data = (float*)malloc((size_t)input_size * sizeof(float*));
		if (input_data != NULL) memset(input_data, 0, input_size);

		// Return a value of 0 if the model loads successfully
		return return_val;
	}

	/// <summary>
	/// Generate object detection proposals from the raw model output
	/// </summary>
	/// <param name="out_ptr">A pointer to the output tensor data</param>
	void GenerateYoloxProposals(float* out_ptr, int proposal_length)
	{
		// Remove the proposals for the previous model output
		proposals.clear();

		// Obtain the number of classes the model was trained to detect
		int num_classes = proposal_length - 5;

		for (int anchor_idx = 0; anchor_idx < grid_strides.size(); anchor_idx++)
		{
			// Get the current grid and stride values
			int grid0 = grid_strides[anchor_idx].grid0;
			int grid1 = grid_strides[anchor_idx].grid1;
			int stride = grid_strides[anchor_idx].stride;

			// Get the starting index for the current proposal
			int start_idx = anchor_idx * proposal_length;

			// Get the coordinates for the center of the predicted bounding box
			float x_center = (out_ptr[start_idx + 0] + grid0) * stride;
			float y_center = (out_ptr[start_idx + 1] + grid1) * stride;

			// Get the dimensions for the predicted bounding box
			float w = exp(out_ptr[start_idx + 2]) * stride;
			float h = exp(out_ptr[start_idx + 3]) * stride;

			// Calculate the coordinates for the upper left corner of the bounding box
			float x0 = x_center - w * 0.5f;
			float y0 = y_center - h * 0.5f;

			// Get the confidence score that an object is present
			float box_objectness = out_ptr[start_idx + 4];

			// Initialize object struct with bounding box information
			Object obj = { x0, y0, w, h, 0, 0 };

			// Find the object class with the highest confidence score
			for (int class_idx = 0; class_idx < num_classes; class_idx++)
			{
				// Get the confidence score for the current object class
				float box_cls_score = out_ptr[start_idx + 5 + class_idx];
				// Calculate the final confidence score for the object proposal
				float box_prob = box_objectness * box_cls_score;

				// Check for the highest confidence score
				if (box_prob > obj.prob)
				{
					obj.label = class_idx;
					obj.prob = box_prob;
				}
			}

			// Only add object proposals with high enough confidence scores
			if (obj.prob > bbox_conf_thresh) proposals.push_back(obj);
		}

		// Sort the proposals based on the confidence score in descending order
		std::sort(proposals.begin(), proposals.end(), [](Object& a, Object& b) -> bool
			{ return a.prob > b.prob; });
	}

	/// <summary>
	/// Filter through a sorted list of object proposals using Non-maximum suppression
	/// </summary>
	void NmsSortedBboxes()
	{
		// Remove the picked proposals for the previous model outptut
		proposal_indices.clear();

		// Iterate through the object proposals
		for (int i = 0; i < proposals.size(); i++)
		{
			Object& a = proposals[i];

			// Create OpenCV rectangle for the Object bounding box
			cv::Rect_<float> rect_a = cv::Rect_<float>(a.x0, a.y0, a.width, a.height);

			bool keep = true;

			// Check if the current object proposal overlaps any selected objects too much
			for (int j : proposal_indices)
			{
				Object& b = proposals[j];

				// Create OpenCV rectangle for the Object bounding box
				cv::Rect_<float> rect_b = cv::Rect_<float>(b.x0, b.y0, b.width, b.height);

				// Calculate the area where the two object bounding boxes overlap
				float inter_area = (rect_a & rect_b).area();
				// Calculate the union area of both bounding boxes
				float union_area = rect_a.area() + rect_b.area() - inter_area;
				// Ignore object proposals that overlap selected objects too much
				if (inter_area / union_area > nms_thresh)
					keep = false;
			}

			// Keep object proposals that do not overlap selected objects too much
			if (keep) proposal_indices.push_back(i);
		}
	}

	/// <summary>
	/// Perform inference with the provided texture data
	/// </summary>
	/// <param name="image_data">The source image data from Unity</param>
	/// <returns>The final number of detected objects</returns>
	DLLExport int PerformInference(uchar* image_data)
	{
		// Store the pixel data for the source input image in an OpenCV Mat
		cv::Mat input_image = cv::Mat(img_h, img_w, CV_8UC4, image_data);
		// Remove the alpha channel
		cv::cvtColor(input_image, input_image, cv::COLOR_RGBA2RGB);
		// Resize the image to the model input dimensions
		cv::resize(input_image, input_image, cv::Size(input_w, input_h));

		// Iterate over each pixel in image
		for (int p = 0; p < n_pixels; p++)
		{
			for (int ch = 0; ch < n_channels; ch++) {
				input_data[ch * n_pixels + p] = ((input_image.data[p * n_channels + ch] / 255.0f) - mean[ch]) / std_dev[ch];
			}
		}

		// Initialize list of input and output names
		const char* input_names[] = { input_name };
		const char* output_names[] = { output_name };
		// Initialize the list of model input dimension
		int64_t input_shape[] = { 1, 3, input_h, input_w };
		int input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);

		// Initialize an input tensor object with the input_data
		OrtMemoryInfo* memory_info;
		ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);

		OrtValue* input_tensor = NULL;
		ort->CreateTensorWithDataAsOrtValue(memory_info, input_data, input_size, input_shape,
			input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
			&input_tensor);

		ort->ReleaseMemoryInfo(memory_info);


		OrtValue* output_tensor = NULL;
		// Perform inference
		ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1,
			&output_tensor);

		// Make sure the output tensor is not NULL to avoid potential crashes
		if (output_tensor == NULL) {
			ort->ReleaseValue(input_tensor);
			ort->ReleaseValue(output_tensor);
			return -1;
		}

		// Get the length of a single object proposal (i.e., number of object classes + 5)
		OrtTensorTypeAndShapeInfo* output_tensor_info;
		ort->GetTensorTypeAndShape(output_tensor, &output_tensor_info);
		size_t output_length[1] = {};
		ort->GetDimensionsCount(output_tensor_info, output_length);
		int64_t output_dims[3] = {};
		ort->GetDimensions(output_tensor_info, output_dims, *output_length);

		// Access model output
		float* out_data;
		ort->GetTensorMutableData(output_tensor, (void**)&out_data);

		// Generate new proposals for the current model output
		GenerateYoloxProposals(out_data, output_dims[2]);

		// Pick detected objects to keep using Non-maximum Suppression
		NmsSortedBboxes();

		// Free memory for input and output tensors
		ort->ReleaseValue(input_tensor);
		ort->ReleaseValue(output_tensor);

		// return the final number of detected objects
		return (int)proposal_indices.size();
	}

	/// <summary>
	/// Fill the provided array with the detected objects
	/// </summary>
	/// <param name="objects">A pointer to a list of objects from Unity</param>
	DLLExport void PopulateObjectsArray(Object* objects)
	{

		for (int i = 0; i < proposal_indices.size(); i++)
		{
			Object obj = proposals[proposal_indices[i]];

			// Adjust offset to source image resolution and clamp the bounding box
			objects[i].x0 = std::min(obj.x0 / scale_x, (float)img_w);
			objects[i].y0 = std::min(obj.y0 / scale_y, (float)img_h);
			objects[i].width = std::min(obj.width / scale_x, (float)img_w);
			objects[i].height = std::min(obj.height / scale_y, (float)img_h);

			objects[i].label = obj.label;
			objects[i].prob = obj.prob;
		}
	}

	/// <summary>
	/// Free memory
	/// </summary>
	DLLExport void FreeResources()
	{
		grid_strides.clear();
		proposals.clear();
		proposal_indices.clear();

		free(input_data);
		ort->ReleaseSession(session);
		ort->ReleaseEnv(env);
	}
}