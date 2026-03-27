package com.example.myapplication


import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Rect
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import org.pytorch.Module
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.io.File
import java.io.FileOutputStream
import org.pytorch.Tensor
import org.pytorch.IValue
import org.pytorch.torchvision.TensorImageUtils





class MainActivity : AppCompatActivity() {


    private val imageSelectCode = 200
    private lateinit var capture: Button
    private lateinit var gallery: Button
    private lateinit var imageView: ImageView
    private lateinit var resultTextView: TextView
    private lateinit var imageClassifier: ImageClassifier
    private lateinit var modelAge: Module
    private var lensFacing = CameraSelector.LENS_FACING_BACK



    private companion object {
        private const val REQUEST_IMAGE_CAPTURE = 1
        private const val REQUEST_PICK_IMAGE = 2
        private const val PERMISSION_REQUEST_CODE = 3
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_my)

        imageClassifier = ImageClassifier(applicationContext,"age_classifier-cpu.pt", "emotion-model.tflite", "gender-model.tflite")
        modelAge = Module.load(assetFilePath(this, "age_classifier-cpu.pt"))


        capture = findViewById(R.id.Capture)
        gallery = findViewById(R.id.gallery)
        imageView = findViewById(R.id.imageView)
        resultTextView = findViewById(R.id.resultTextView)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED
            || ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
            || ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {

            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE), PERMISSION_REQUEST_CODE)
        }



        capture.setOnClickListener {
            val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            if (takePictureIntent.resolveActivity(packageManager) != null) {
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
            }
        }

        gallery.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(intent, REQUEST_PICK_IMAGE)
        }
    }





    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == Activity.RESULT_OK) {
            when (requestCode) {
                REQUEST_IMAGE_CAPTURE -> {
                    var imageBitmap = data?.extras?.get("data") as? Bitmap
                    if (imageBitmap != null) {
                        imageBitmap = resizeBitmap(imageBitmap, 480)
                        imageView.setImageBitmap(imageBitmap)
                        Log.d("Bitmap Size", "${imageBitmap.width} x ${imageBitmap.height}")

                        // Show loading state
                        resultTextView.visibility = View.VISIBLE
                        resultTextView.text = "Detecting face and analyzing..."

                        // Use callback to handle results
                        imageClassifier.classifyImage(imageBitmap) { ageRange, emotion, gender ->
                            Log.d("Results", "Age: $ageRange, Emotion: $emotion, Gender: $gender")
                            resultTextView.text = "Age: $ageRange, Emotion: $emotion, Gender: $gender"
                        }
                    }
                }
                REQUEST_PICK_IMAGE -> {
                    // Gallery pick returns a URI
                    data?.data?.let { uri ->
                        processImage(uri)
                    }
                }
            }
        }
    }

    private fun resizeBitmap(bitmap: Bitmap, maxSize: Int): Bitmap {
        val ratio = minOf(
            maxSize.toFloat() / bitmap.width,
            maxSize.toFloat() / bitmap.height
        )

        val newWidth = (bitmap.width * ratio).toInt()
        val newHeight = (bitmap.height * ratio).toInt()

        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
    }



    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)

        if (!file.exists()) {
            context.assets.open(assetName).use { input ->
                FileOutputStream(file).use { output ->
                    input.copyTo(output)
                }
            }
        }

        return file.absolutePath
    }
    


    private fun processImage(imageUri: Uri) {
        try {
            var bitmap = MediaStore.Images.Media.getBitmap(contentResolver, imageUri)
            if (bitmap != null) {
                bitmap = resizeBitmap(bitmap, 480)
                imageView.setImageBitmap(bitmap)
                Log.d("Bitmap Size", "${bitmap.width} x ${bitmap.height}")
                // Show loading state
                resultTextView.text = "Detecting face and analyzing..."
                resultTextView.visibility = View.VISIBLE
                // Run classification (it's now async)
                imageClassifier.classifyImage(bitmap) { ageRange, emotion, gender ->
                    // This callback runs when results are ready
                    Log.d("Results", "Age: $ageRange, Emotion: $emotion, Gender: $gender")
                    resultTextView.text = "Age: $ageRange, Emotion: $emotion, Gender: $gender"
                }
            } else {
                resultTextView.text = "Failed to load image."
            }
        } catch (e: Exception) {
            Log.e("ProcessImage", "Error processing image", e)
            resultTextView.text = "Error: ${e.message}"
        }
    }



    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Permissions granted", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Permissions denied", Toast.LENGTH_SHORT).show()
            }
        }
    }
}




class ImageClassifier(context: Context, modelPathAge : String, modelPathEmotion: String, modelPathGender: String) {

    private val modelAge = Module.load(assetFilePath(context, "age_classifier-cpu.pt"))
    private val modelEmotion: Interpreter = Interpreter(loadModelFile(context, "emotion-model.tflite"))
    private val modelGender: Interpreter = Interpreter(loadModelFile(context, "gender-model.tflite"))
    private val faceDetector: FaceDetector = FaceDetection.getClient()




    private fun loadModelFile(context:Context, modelPath: String): ByteBuffer {
        val assetManager = context.assets
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun classifyImage(image: Bitmap, callback: (String, String, String) -> Unit) {
        val inputImage = InputImage.fromBitmap(image, 0)

        faceDetector.process(inputImage)
            .addOnSuccessListener { faces ->
                if (faces.isEmpty()) {
                    Log.w("FaceDetection", "No faces detected in image")
                    callback("N/A", "N/A", "N/A") // Return N/A if no face
                    return@addOnSuccessListener
                }

                val face = faces[0]
                val croppedFace = cropBitmap(image, face.boundingBox)
                val result = runClassification(croppedFace)
                callback(result.first, result.second, result.third)
            }
            .addOnFailureListener { e ->
                Log.e("FaceDetection", "Face detection failed", e)
                callback("Error", "Error", "Error")
            }
    }


    private fun cropBitmap(bitmap: Bitmap, boundingBox: Rect): Bitmap {
        // Add padding around the face for better results
        val padding = 20
        val left = maxOf(0, boundingBox.left - padding)
        val top = maxOf(0, boundingBox.top - padding)
        val width = minOf(boundingBox.width() + padding * 2, bitmap.width - left)
        val height = minOf(boundingBox.height() + padding * 2, bitmap.height - top)

        return Bitmap.createBitmap(bitmap, left, top, width, height)
    }


    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)

        if (!file.exists()) {
            context.assets.open(assetName).use { input ->
                FileOutputStream(file).use { output ->
                    input.copyTo(output)
                }
            }
        }

        return file.absolutePath
    }


    fun bitmapToByteBuffer(bitmap: Bitmap, inputSize: Int, isGrayscale: Boolean): ByteBuffer {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val byteBuffer = ByteBuffer.allocateDirect(
            if (isGrayscale) {
                inputSize * inputSize * 4 // 1 channel × 4 bytes per float
            } else {
                inputSize * inputSize * 3 * 4 // 3 channels (RGB) × 4 bytes per float
            }
        )
        byteBuffer.order(ByteOrder.nativeOrder())
        byteBuffer.rewind()

        val pixels = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(pixels, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)

        for (pixel in pixels) {
            if (isGrayscale) {
                // Extract single grayscale value
                val gray = (pixel shr 16) and 0xFF
                byteBuffer.putFloat(gray / 255.0f)
            } else {
                // Extract RGB values
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                byteBuffer.putFloat(r / 255.0f)
                byteBuffer.putFloat(g / 255.0f)
                byteBuffer.putFloat(b / 255.0f)
            }
        }

        byteBuffer.rewind()
        return byteBuffer
    }


    fun runClassification(image: Bitmap): Triple<String, String, String> {

        val ageInput = Bitmap.createScaledBitmap(image, 128, 128, true)
        val genderInput = Bitmap.createScaledBitmap(image, 100, 100, true)
        val emotionInput = Bitmap.createScaledBitmap(image, 200, 200, true)

        // ===== AGE MODEL (PyTorch) =====
        val ageTensor = TensorImageUtils.bitmapToFloat32Tensor(
            ageInput,
            floatArrayOf(0.485f, 0.456f, 0.406f),
            floatArrayOf(0.229f, 0.224f, 0.225f)
        )

        val ageOutput = modelAge.forward(IValue.from(ageTensor)).toTensor()
        val ageScores = ageOutput.dataAsFloatArray
        Log.d("Age Result", ageScores.contentToString())

        val ageIndex = ageScores.indices.maxByOrNull { ageScores[it] } ?: 0
        val age = getAgeRange(ageIndex)
        val genderInputBuffer = bitmapToByteBuffer(genderInput, 100, isGrayscale = true)
        val emotionInputBuffer = bitmapToByteBuffer(emotionInput, 200, isGrayscale = true)

        val ageResult = Array(1) { FloatArray(1) }
        val emotionResult = Array(1) { FloatArray(6) }
        val genderResult = Array(1) { FloatArray(2) }

        modelEmotion.run(emotionInputBuffer, emotionResult)
        modelGender.run(genderInputBuffer, genderResult)

        Log.d("Age Result", ageResult[0].contentToString())
        Log.d("Emotion Result", emotionResult[0].contentToString())
        Log.d("Gender Result", genderResult[0].contentToString())

        val emotionIndex = getEmotionFromResult(emotionResult[0])
        val genderIndex = if (genderResult[0][0] > 0.5) "Male" else "Female"

        return Triple(age, emotionIndex, genderIndex)
    }


    private fun getMaxIndex(array: FloatArray): Int {
        var maxIndex = 0
        var maxValue = array[0]
        for (i in array.indices) {
            if (array[i] > maxValue) {
                maxValue = array[i]
                maxIndex = i
            }
        }
        return maxIndex
    }

    private fun getAgeRange(ageGroupIndex: Int): String {
        return when (ageGroupIndex) {
            0 -> "child (0-10)"
            1 -> "teen (11-20)"
            2 -> "young (21-30)"
            3 -> "middle (31-40)"
            4 -> "mature (41-50)"
            5 -> "senior (51-60)"
            6 -> "older (61-70)"
            7 -> "elderly (71+)"
            else -> "Unknown"
        }
    }

    private fun getEmotionFromResult(results: FloatArray): String {
        val emotionLabels = arrayOf("ahego", "angry", "happy", "neutral", "sad", "surprise")
        // Find index of maximum value
        var emotionIndex = 0
        var maxValue = results[0]

        for (i in results.indices) {
            if (results[i] > maxValue) {
                maxValue = results[i]
                emotionIndex = i
            }
        }

        return emotionLabels[emotionIndex]
    }

    fun close() {
        //model_Age.close()
        modelEmotion.close()
        modelGender.close()
    }
}



