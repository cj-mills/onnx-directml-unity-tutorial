using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using System;
using UnityEngine.UI;
using System.Runtime.InteropServices;
using System.IO;

#if UNITY_EDITOR
using UnityEditor;


[InitializeOnLoad]
public class Startup
{
    static Startup()
    {
        // Get all files named "DirectML.dll"
        string[] files = Directory.GetFiles("./Assets/", "DirectML.dll", SearchOption.AllDirectories);
        // Iterate through each found file
        foreach (string file in files)
        {
            // Check if the file is in the "x86_64" folder
            if (file.Contains("x86_64"))
            {
                // Define file path for the Editor application directory
                string editorPath = EditorApplication.applicationPath;
                string editorDir = Directory.GetParent(editorPath).ToString();
                Debug.Log($"Editor Directory: {editorDir}");
                string targetPath = $"{editorDir}/DirectML.dll";
                // Only copy the file to the StreamingAssets folder if it is not already present
                if (!File.Exists(targetPath)) File.Copy(file, targetPath);
            }
        }
    }
}
#endif

public class ObjectDetector : MonoBehaviour
{

    [Header("Scene Objects")]
    [Tooltip("The Screen object for the scene")]
    public Transform screen;
    [Tooltip("Mirror the in-game screen.")]
    public bool mirrorScreen = true;

    [Header("Data Processing")]
    [Tooltip("The target minimum model input dimensions")]
    public int targetDim = 224;
    [Tooltip("The compute shader for GPU processing")]
    public ComputeShader processingShader;
    [Tooltip("Asynchronously download input image from the GPU to the CPU.")]
    public bool useAsyncGPUReadback = true;

    [Header("Output Processing")]
    [Tooltip("A json file containing the colormaps for object classes")]
    public TextAsset colormapFile;
    [Tooltip("Minimum confidence score for keeping detected objects")]
    [Range(0, 1f)]
    public float minConfidence = 0.5f;

    [Header("Debugging")]
    [Tooltip("Print debugging messages to the console")]
    public bool printDebugMessages = true;

    [Header("Webcam")]
    [Tooltip("Use a webcam as input")]
    public bool useWebcam = false;
    [Tooltip("The requested webcam dimensions")]
    public Vector2Int webcamDims = new Vector2Int(1280, 720);
    [Tooltip("The requested webcam framerate")]
    [Range(0, 60)]
    public int webcamFPS = 60;

    [Header("GUI")]
    [Tooltip("Display predicted class")]
    public bool displayBoundingBoxes = true;
    [Tooltip("Display number of detected objects")]
    public bool displayProposalCount = true;
    [Tooltip("Display fps")]
    public bool displayFPS = true;
    [Tooltip("The on-screen text color")]
    public Color textColor = Color.red;
    [Tooltip("The scale value for the on-screen font size")]
    [Range(0, 99)]
    public int fontScale = 50;
    [Tooltip("The number of seconds to wait between refreshing the fps value")]
    [Range(0.01f, 1.0f)]
    public float fpsRefreshRate = 0.1f;
    [Tooltip("The toggle for using a webcam as the input source")]
    public Toggle useWebcamToggle;
    [Tooltip("The dropdown menu that lists available webcam devices")]
    public Dropdown webcamDropdown;
    [Tooltip("The dropdown menu that lists available ONNX models")]
    public Dropdown modelDropdown;
    [Tooltip("The dropdown menu that lists available ONNX execution providers")]
    public Dropdown executionProviderDropdown;

    [Header("ONNX")]
    [Tooltip("The name of the ONNX models folder")]
    public string onnxModelsDir = "ONNXModels";

    // List of available webcam devices
    private WebCamDevice[] webcamDevices;
    // Live video input from a webcam
    private WebCamTexture webcamTexture;
    // The name of the current webcam  device
    private string currentWebcam;

    // The test image dimensions
    private Vector2Int imageDims;
    // The test image texture
    private Texture imageTexture;
    // The current screen object dimensions
    private Vector2Int screenDims;
    // The model GPU input texture
    private RenderTexture inputTextureGPU;
    // The model CPU input texture
    private Texture2D inputTextureCPU;

    // Stores the number of detected objects
    private int numObjects;

    // A class for parsing in colormaps from a JSON file
    [System.Serializable]
    class ColorMap { public string label; public float[] color; }
    // A class for reading in a list of colormaps from a JSON file
    [System.Serializable]
    class ColorMapList { public List<ColorMap> items; }
    // Stores a list of colormaps from a JSON file
    private ColorMapList colormapList;
    // A list of colors that map to class labels
    private Color[] colors;
    // A list of single pixel textures that map to class labels
    private Texture2D[] colorTextures;

    // The current frame rate value
    private int fps = 0;
    // Controls when the frame rate value updates
    private float fpsTimer = 0f;

    // File paths for the available ONNX models
    private List<string> modelPaths = new List<string>();
    // Names of the available ONNX models
    private List<string> modelNames = new List<string>();
    // Names of the available ONNX execution providers
    private List<string> onnxExecutionProviders = new List<string>();

    
    // Indicate that the members of the struct are laid out sequentially
    [StructLayout(LayoutKind.Sequential)]
    /// <summary>
    /// Stores the information for a single object
    /// </summary> 
    public struct Object
    {
        // The X coordinate for the top left bounding box corner
        public float x0;
        // The Y coordinate for the top left bounding box cornder
        public float y0;
        // The width of the bounding box
        public float width;
        // The height of the bounding box
        public float height;
        // The object class index for the detected object
        public int label;
        // The model confidence score for the object
        public float prob;

        public Object(float x0, float y0, float width, float height, int label, float prob)
        {
            this.x0 = x0;
            this.y0 = y0;
            this.width = width;
            this.height = height;
            this.label = label;
            this.prob = prob;
        }
    }

    // Stores information for the current list of detected objects
    private Object[] objectInfoArray;

    // Name of the DLL file
    const string dll = "ONNX_YOLOX_DLL";

    [DllImport(dll)]
    private static extern int InitOrtAPI();

    [DllImport(dll)]
    private static extern int GetProviderCount();

    [DllImport(dll)]
    private static extern IntPtr GetProviderName(int index);

    [DllImport(dll)]
    private static extern void SetConfidenceThreshold(float minConfidence);

    [DllImport(dll)]
    private static extern void RefreshMemory();

    [DllImport(dll)]
    private static extern int LoadModel(string model, string execution_provider, int[] inputDims);

    [DllImport(dll)]
    private static extern int PerformInference(IntPtr inputData);

    [DllImport(dll)]
    private static extern void PopulateObjectsArray(IntPtr objects);

    [DllImport(dll)]
    private static extern void FreeResources();


    /// <summary>
    /// Initialize the selected webcam device
    /// </summary>
    /// <param name="deviceName">The name of the selected webcam device</param>
    private void InitializeWebcam(string deviceName)
    {
        // Stop any webcams already playing
        if (webcamTexture && webcamTexture.isPlaying) webcamTexture.Stop();

        // Create a new WebCamTexture
        webcamTexture = new WebCamTexture(deviceName, webcamDims.x, webcamDims.y, webcamFPS);

        // Start the webcam
        webcamTexture.Play();
        // Check if webcam is playing
        useWebcam = webcamTexture.isPlaying;
        // Update toggle value
        useWebcamToggle.SetIsOnWithoutNotify(useWebcam);

        Debug.Log(useWebcam ? "Webcam is playing" : "Webcam not playing, option disabled");
    }

    /// <summary>
    /// Resize and position an in-scene screen object
    /// </summary>
    private void InitializeScreen()
    {
        // Set the texture for the screen object
        screen.gameObject.GetComponent<MeshRenderer>().material.mainTexture = useWebcam ? webcamTexture : imageTexture;
        // Set the screen dimensions
        screenDims = useWebcam ? new Vector2Int(webcamTexture.width, webcamTexture.height) : imageDims;

        // Flip the screen around the Y-Axis when using webcam
        float yRotation = useWebcam && mirrorScreen ? 180f : 0f;
        // Invert the scale value for the Z-Axis when using webcam
        float zScale = useWebcam && mirrorScreen ? -1f : 1f;

        // Set screen rotation
        screen.rotation = Quaternion.Euler(0, yRotation, 0);
        // Adjust the screen dimensions
        screen.localScale = new Vector3(screenDims.x, screenDims.y, zScale);

        // Adjust the screen position
        screen.position = new Vector3(screenDims.x / 2, screenDims.y / 2, 1);
    }

    /// <summary>
    /// Get the file paths for available ONNX models
    /// </summary>
    private void GetONNXModels()
    {
        // Get the paths for each model folder
        foreach (string dir in System.IO.Directory.GetDirectories($"{Application.streamingAssetsPath}/{onnxModelsDir}"))
        {
            string modelName = dir.Split('\\')[1];

            modelNames.Add(modelName.Substring(0, modelName.Length));

            // Get the paths for the ONNX file for each model
            foreach (string file in System.IO.Directory.GetFiles(dir))
            {
                if (file.EndsWith(".onnx"))
                {
                    modelPaths.Add(file);
                }
            }
        }
    }

    /// <summary>
    /// Get the names of the available ONNX execution providers
    /// </summary>
    private void GetONNXExecutionProviders()
    {
        // Get the number of available ONNX execution providers
        int providerCount = GetProviderCount();
        Debug.Log($"Provider Count: {providerCount}");

        for (int i = 0; i < providerCount; i++)
        {
            string providerName = Marshal.PtrToStringAnsi(GetProviderName(i));
            Debug.Log(providerName);
            providerName = providerName.Replace("ExecutionProvider", "");
            onnxExecutionProviders.Add(providerName);
        }
        onnxExecutionProviders.Reverse();
    }

    /// <summary>
    /// Initialize the GUI dropdown list
    /// </summary>
    private void InitializeDropdown()
    {
        // Create list of webcam device names
        List<string> webcamNames = new List<string>();
        foreach (WebCamDevice device in webcamDevices) webcamNames.Add(device.name);

        // Remove default dropdown options
        webcamDropdown.ClearOptions();
        // Add webcam device names to dropdown menu
        webcamDropdown.AddOptions(webcamNames);
        // Set the value for the dropdown to the current webcam device
        webcamDropdown.SetValueWithoutNotify(webcamNames.IndexOf(currentWebcam));

        // Remove default dropdown options
        modelDropdown.ClearOptions();
        // Add ONNX model names to menu
        modelDropdown.AddOptions(modelNames);
        // Select the first option in the dropdown
        modelDropdown.SetValueWithoutNotify(0);

        // Remove default dropdown options
        executionProviderDropdown.ClearOptions();
        // Add ONNX provider names to menu
        executionProviderDropdown.AddOptions(onnxExecutionProviders);
        // Select the first option in the dropdown
        executionProviderDropdown.SetValueWithoutNotify(0);
    }

    /// <summary>
    /// Resize and position the main camera based on an in-scene screen object
    /// </summary>
    /// <param name="screenDims">The dimensions of an in-scene screen object</param>
    private void InitializeCamera(Vector2Int screenDims, string cameraName = "Main Camera")
    {
        // Get a reference to the Main Camera GameObject
        GameObject camera = GameObject.Find(cameraName);
        // Adjust the camera position to account for updates to the screenDims
        camera.transform.position = new Vector3(screenDims.x / 2, screenDims.y / 2, -10f);
        // Render objects with no perspective (i.e. 2D)
        camera.GetComponent<Camera>().orthographic = true;
        // Adjust the camera size to account for updates to the screenDims
        camera.GetComponent<Camera>().orthographicSize = screenDims.y / 2;
    }

    /// <summary>
    /// Update the selected ONNX model
    /// </summary>
    public void UpdateONNXModel()
    {
        // Reset objectInfoArray
        objectInfoArray = new Object[0];

        int[] inputDims = new int[] {
            inputTextureCPU.width,
            inputTextureCPU.height
        };

        Debug.Log($"Source input dims: {inputDims[0]} x {inputDims[1]}");

        // Load the specified ONNX model
        int return_msg = LoadModel(
            modelPaths[modelDropdown.value], 
            onnxExecutionProviders[executionProviderDropdown.value], 
            inputDims);

        SetConfidenceThreshold(minConfidence);

        string[] return_messages = {
            "Using DirectML",
            "Using CPU",
        };

        Debug.Log($"Updated input dims: {inputDims[0]} x {inputDims[1]}");
        Debug.Log($"Return message: {return_messages[return_msg]}");
    }

    // Awake runs when the script instance is being loaded
    private void Awake()
    {
#if !UNITY_EDITOR
        // Define the path for the DirectML.dll file in the StreamingAssets folder
        string sourcePath = $"{Application.dataPath}/Plugins/x86_64/DirectML.dll";

        string dataPath = Application.dataPath;
        string buildDir = Directory.GetParent(dataPath).ToString();

        // Define the destination path for the DirectML.dll file
        string targetPath = $"{buildDir}/DirectML.dll";
        // Only copy the file if it is not already present at the destination
        if (!File.Exists(targetPath)) File.Copy(sourcePath, targetPath);
#endif
    }

    // Start runs before the first frame update
    void Start()
    {
        // Get the source image texture
        imageTexture = screen.gameObject.GetComponent<MeshRenderer>().material.mainTexture;
        // Get the source image dimensions as a Vector2Int
        imageDims = new Vector2Int(imageTexture.width, imageTexture.height);

        // Initialize list of available webcam devices
        webcamDevices = WebCamTexture.devices;
        foreach (WebCamDevice device in webcamDevices) Debug.Log(device.name);
        currentWebcam = webcamDevices[0].name;
        useWebcam = webcamDevices.Length > 0 ? useWebcam : false;
        // Initialize webcam
        if (useWebcam) InitializeWebcam(currentWebcam);

        // Resize and position the screen object using the source image dimensions
        InitializeScreen();
        // Resize and position the main camera using the source image dimensions
        InitializeCamera(screenDims);

        // Initialize list of color maps from JSON file
        colormapList = JsonUtility.FromJson<ColorMapList>(colormapFile.text);
        // Initialize the list of colors
        colors = new Color[colormapList.items.Count];
        // Initialize the list of color textures
        colorTextures = new Texture2D[colormapList.items.Count];

        // Populate the color and color texture arrays
        for (int i = 0; i < colors.Length; i++)
        {
            // Create a new color object
            colors[i] = new Color(
                colormapList.items[i].color[0],
                colormapList.items[i].color[1],
                colormapList.items[i].color[2]);
            // Create a single-pixel texture
            colorTextures[i] = new Texture2D(1, 1);
            colorTextures[i].SetPixel(0, 0, colors[i]);
            colorTextures[i].Apply();

        }

        // Get the file paths for available ONNX models
        GetONNXModels();

        InitOrtAPI();
        // Get the names of available ONNX execution providers
        GetONNXExecutionProviders();

        // Initialize the webcam dropdown list
        InitializeDropdown();
    }


    /// <summary>
    /// Process the provided image using the specified function on the GPU
    /// </summary>
    /// <param name="image">The target image RenderTexture</param>
    /// <param name="computeShader">The target ComputerShader</param>
    /// <param name="functionName">The target ComputeShader function</param>
    /// <returns></returns>
    private void ProcessImageGPU(RenderTexture image, ComputeShader computeShader, string functionName)
    {
        // Specify the number of threads on the GPU
        int numthreads = 8;
        // Get the index for the specified function in the ComputeShader
        int kernelHandle = computeShader.FindKernel(functionName);
        // Define a temporary HDR RenderTexture
        RenderTexture result = new RenderTexture(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);
        // Enable random write access
        result.enableRandomWrite = true;
        // Create the HDR RenderTexture
        result.Create();

        // Set the value for the Result variable in the ComputeShader
        computeShader.SetTexture(kernelHandle, "Result", result);
        // Set the value for the InputImage variable in the ComputeShader
        computeShader.SetTexture(kernelHandle, "InputImage", image);

        // Execute the ComputeShader
        computeShader.Dispatch(kernelHandle, result.width / numthreads, result.height / numthreads, 1);

        // Copy the result into the source RenderTexture
        Graphics.Blit(result, image);

        // Release RenderTexture
        result.Release();
    }


    /// <summary>
    /// Scale the source image resolution to the target input dimensions
    /// while maintaing the source aspect ratio.
    /// </summary>
    /// <param name="imageDims"></param>
    /// <param name="targetDims"></param>
    /// <returns></returns>
    private Vector2Int CalculateInputDims(Vector2Int imageDims, int targetDim)
    {
        Vector2Int inputDims = new Vector2Int();

        // Calculate the input dimensions using the target minimum dimension
        if (imageDims.x >= imageDims.y)
        {
            inputDims[0] = (int)(imageDims.x / ((float)imageDims.y / (float)targetDim));
            inputDims[1] = targetDim;
        }
        else
        {
            inputDims[0] = targetDim;
            inputDims[1] = (int)(imageDims.y / ((float)imageDims.x / (float)targetDim));
        }

        return inputDims;
    }


    /// <summary>
    /// Called once AsyncGPUReadback has been completed
    /// </summary>
    /// <param name="request"></param>
    private void OnCompleteReadback(AsyncGPUReadbackRequest request)
    {
        if (request.hasError)
        {
            Debug.Log("GPU readback error detected.");
            return;
        }

        // Make sure the Texture2D is not null
        if (inputTextureCPU)
        {
            // Fill Texture2D with raw data from the AsyncGPUReadbackRequest
            inputTextureCPU.LoadRawTextureData(request.GetData<uint>());
            // Apply changes to Textur2D
            inputTextureCPU.Apply();
        }
    }


    /// <summary>
    /// Pin memory for the input data and pass a reference to the plugin for inference
    /// </summary>
    /// <param name="texture">The input texture</param>
    /// <returns></returns>
    public unsafe int UploadTexture(Texture2D texture)
    {
        //Pin Memory
        fixed (byte* p = texture.GetRawTextureData())
        {
            // Perform inference and get the number of detected objects
            numObjects = PerformInference((IntPtr)p);
        }

        if (numObjects == -1)
        {
            Debug.Log("Failed to populate output_tensor in the PerformInference function.");
            numObjects = 0;
        }

        // Initialize the array
        objectInfoArray = new Object[numObjects];

        // Pin memory
        fixed (Object* o = objectInfoArray)
        {
            // Get the detected objects
            PopulateObjectsArray((IntPtr)o);
        }

        return numObjects;
    }


    /// <summary>
    /// Scale the latest bounding boxes to the display resolution
    /// </summary>
    public void ScaleBoundingBoxes()
    {
        // Process new detected objects
        for (int i = 0; i < objectInfoArray.Length; i++)
        {
            // The smallest dimension of the screen
            float minScreenDim = Mathf.Min(screen.transform.localScale.x, screen.transform.localScale.y);
            // 
            int minInputDim = Mathf.Min(inputTextureCPU.width, inputTextureCPU.height);
            // 
            float minImgScale = minScreenDim / minInputDim;
            // 
            float displayScale = Screen.height / screen.transform.localScale.y;

            // Scale bounding box to in-game screen resolution and flip the bbox coordinates vertically
            float x0 = objectInfoArray[i].x0 * minImgScale;
            float y0 = (inputTextureCPU.height - objectInfoArray[i].y0) * minImgScale;
            float width = objectInfoArray[i].width * minImgScale;
            float height = objectInfoArray[i].height * minImgScale;

            // 
            if (mirrorScreen && useWebcam) x0 = screen.transform.localScale.x - x0 - width;

            // Scale bounding boxes to display resolution
            objectInfoArray[i].x0 = x0 * displayScale;
            objectInfoArray[i].y0 = y0 * displayScale;
            objectInfoArray[i].width = width * displayScale;
            objectInfoArray[i].height = height * displayScale;

            // Offset the bounding box coordinates based on the difference between the in-game screen and display
            objectInfoArray[i].x0 += (Screen.width - screen.transform.localScale.x * displayScale) / 2;
        }
    }



    // Update runs once per frame
    void Update()
    {
        useWebcam = webcamDevices.Length > 0 ? useWebcam : false;
        if (useWebcam)
        {
            // Initialize webcam if it is not already playing
            if (!webcamTexture || !webcamTexture.isPlaying) InitializeWebcam(currentWebcam);

            // Skip the rest of the method if the webcam is not initialized
            if (webcamTexture.width <= 16) return;

            // Make sure screen dimensions match webcam resolution when using webcam
            if (screenDims.x != webcamTexture.width)
            {
                // Resize and position the screen object using the source image dimensions
                InitializeScreen();
                // Resize and position the main camera using the source image dimensions
                InitializeCamera(screenDims);
            }
        }
        else if (webcamTexture && webcamTexture.isPlaying)
        {
            // Stop the current webcam
            webcamTexture.Stop();

            // Resize and position the screen object using the source image dimensions
            InitializeScreen();
            // Resize and position the main camera using the source image dimensions
            InitializeCamera(screenDims);
        }

        // Scale the source image resolution
        Vector2Int inputDims = CalculateInputDims(screenDims, targetDim);
        if (printDebugMessages) Debug.Log($"Input Dims: {inputDims.x} x {inputDims.y}");

        // Initialize the input texture with the calculated input dimensions
        inputTextureGPU = RenderTexture.GetTemporary(inputDims.x, inputDims.y, 24, RenderTextureFormat.ARGBHalf);

        if (!inputTextureCPU || inputTextureCPU.width != inputTextureGPU.width)
        {
            inputTextureCPU = new Texture2D(inputDims.x, inputDims.y, TextureFormat.RGBA32, false);
            // Update the selected ONNX model
            UpdateONNXModel();
        }

        // Copy the source texture into model input texture
        Graphics.Blit((useWebcam ? webcamTexture : imageTexture), inputTextureGPU);

        // Flip image before sending to DLL
        ProcessImageGPU(inputTextureGPU, processingShader, "FlipXAxis");

        // Download pixel data from GPU to CPU
        if (useAsyncGPUReadback)
        {
            AsyncGPUReadback.Request(inputTextureGPU, 0, TextureFormat.RGBA32, OnCompleteReadback);
        }
        else
        {
            RenderTexture.active = inputTextureGPU;
            inputTextureCPU.ReadPixels(new Rect(0, 0, inputTextureGPU.width, inputTextureGPU.height), 0, 0);
            inputTextureCPU.Apply();
        }

        // Send reference to inputData to DLL
        numObjects = UploadTexture(inputTextureCPU);
        if (printDebugMessages) Debug.Log($"Detected {numObjects} objects");
        // Scale bounding boxes
        ScaleBoundingBoxes();

        // Release the input texture
        RenderTexture.ReleaseTemporary(inputTextureGPU);
    }

    /// <summary>
    /// This method runs when the value for an ONNX dropdown changes
    /// </summary>
    public void ONNXDropdownUpdate()
    {
        // Only call plugin methods after initializing the input texture
        if (inputTextureCPU)
        {
            RefreshMemory();
            UpdateONNXModel();
        }
    }

    /// <summary>
    /// This method runs when the value for the webcam toggle changes
    /// </summary>
    /// <param name="useWebcam"></param>
    public void UpdateWebcamToggle(bool useWebcam)
    {
        this.useWebcam = useWebcam;
    }

    /// <summary>
    /// The method runs when the selected value for the webcam dropdown changes
    /// </summary>
    public void UpdateWebcamDevice()
    {
        currentWebcam = webcamDevices[webcamDropdown.value].name;
        Debug.Log($"Selected Webcam: {currentWebcam}");
        // Initialize webcam if it is not already playing
        if (useWebcam) InitializeWebcam(currentWebcam);

        // Resize and position the screen object using the source image dimensions
        InitializeScreen();
        // Resize and position the main camera using the source image dimensions
        InitializeCamera(screenDims);
    }

    /// <summary>
    /// Update the minimum confidence score for keeping bounding box proposals
    /// </summary>
    /// <param name="slider"></param>
    public void UpdateConfidenceThreshold(Slider slider)
    {
        minConfidence = slider.value;
        SetConfidenceThreshold(minConfidence);
    }

    // OnGUI handles and renders GUI events.
    public void OnGUI()
    {
        // Initialize a rectangle for label text
        Rect labelRect = new Rect();
        // Initialize a rectangle for bounding boxes
        Rect boxRect = new Rect();

        GUIStyle labelStyle = new GUIStyle
        {
            fontSize = (int)(Screen.width * 11e-3)
        };
        labelStyle.alignment = TextAnchor.MiddleLeft;

        foreach (Object objectInfo in objectInfoArray)
        {
            if (!displayBoundingBoxes) break;

            // Skip object if label index is out of bounds
            if (objectInfo.label > colors.Length - 1) continue;

            // Get color for current class index
            Color color = colors[objectInfo.label];
            // Get label for current class index
            string name = colormapList.items[objectInfo.label].label;

            // Set bounding box coordinates
            boxRect.x = objectInfo.x0;
            boxRect.y = Screen.height - objectInfo.y0;
            // Set bounding box dimensions
            boxRect.width = objectInfo.width;
            boxRect.height = objectInfo.height;

            // Scale bounding box line width based on display resolution
            int lineWidth = (int)(Screen.width * 1.75e-3);
            // Render bounding box
            GUI.DrawTexture(
                position: boxRect,
                image: Texture2D.whiteTexture,
                scaleMode: ScaleMode.StretchToFill,
                alphaBlend: true,
                imageAspect: 0,
                color: color,
                borderWidth: lineWidth,
                borderRadius: 0);

            // Include class label and confidence score in label text
            string labelText = $" {name}: {(objectInfo.prob * 100).ToString("0.##")}%";

            // Initialize label GUI content
            GUIContent labelContent = new GUIContent(labelText);

            // Calculate the text size.
            Vector2 textSize = labelStyle.CalcSize(labelContent);

            // Set label text coordinates
            labelRect.x = objectInfo.x0;
            labelRect.y = Screen.height - objectInfo.y0 - textSize.y + lineWidth;

            // Set label text dimensions
            labelRect.width = Mathf.Max(textSize.x, objectInfo.width);
            labelRect.height = textSize.y;
            // Set label text and backgound color
            labelStyle.normal.textColor = color.grayscale > 0.5 ? Color.black : Color.white;
            labelStyle.normal.background = colorTextures[objectInfo.label];
            // Render label
            GUI.Label(labelRect, labelContent, labelStyle);

            Rect objectDot = new Rect();
            objectDot.height = lineWidth * 5;
            objectDot.width = lineWidth * 5;
            float radius = objectDot.width / 2;
            objectDot.x = (boxRect.x + boxRect.width / 2) - radius;
            objectDot.y = (boxRect.y + boxRect.height / 2) - radius;


            GUI.DrawTexture(
                position: objectDot,
                image: Texture2D.whiteTexture,
                scaleMode: ScaleMode.StretchToFill,
                alphaBlend: true,
                imageAspect: 0,
                color: color,
                borderWidth: radius,
                borderRadius: radius);

        }

        // Define styling information for GUI elements
        GUIStyle style = new GUIStyle
        {
            fontSize = (int)(Screen.width * (1f / (100f - fontScale)))
        };
        style.normal.textColor = textColor;

        // Define screen spaces for GUI elements
        Rect slot1 = new Rect(10, 10, 500, 500);
        Rect slot2 = new Rect(10, style.fontSize * 1.5f, 500, 500);

        string content = $"Objects Detected: {numObjects}";
        if (displayProposalCount) GUI.Label(slot1, new GUIContent(content), style);

        // Update framerate value
        if (Time.unscaledTime > fpsTimer)
        {
            fps = (int)(1f / Time.unscaledDeltaTime);
            fpsTimer = Time.unscaledTime + fpsRefreshRate;
        }

        // Adjust screen position when not showing predicted class
        Rect fpsRect = displayProposalCount ? slot2 : slot1;
        if (displayFPS) GUI.Label(fpsRect, new GUIContent($"FPS: {fps}"), style);
    }

    /// <summary>
    /// This method runs when the user clicks the GUI Quit button
    /// </summary>
    public void Quit()
    {
        Application.Quit();
    }

    private void OnApplicationQuit()
    {
        FreeResources();
    }
}
