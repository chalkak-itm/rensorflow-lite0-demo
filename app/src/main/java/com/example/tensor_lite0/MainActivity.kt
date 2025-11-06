package com.example.tensor_lite0

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.result.registerForActivityResult
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.ObjectDetector

class MainActivity : AppCompatActivity() {
    private lateinit var imageView: ImageView
    private lateinit var textView: TextView
    private lateinit var buttonSelect: Button
    private lateinit var detector: ObjectDetector

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageViewResult)
        textView = findViewById(R.id.textViewResult)
        buttonSelect = findViewById(R.id.buttonSelectImage)

        // Initial the model
        detector = createObjectDetector()

        // Choosing image in local gallery
        val imagePicker = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result ->
            if (result.resultCode == Activity.RESULT_OK && result.data != null) {
                val uri: Uri? = result.data!!.data
                uri?.let {
                    val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, it)
                    runObjectDetection(bitmap)
                }
            }
        }
        // Open Gallery when click button
        buttonSelect.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            imagePicker.launch(intent)
        }
    }
    // Bring EfficientDet Lite0 model
    private fun createObjectDetector(): ObjectDetector {
        val baseOptions = BaseOptions.builder()
            .setNumThreads(4)
            .build()

        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setScoreThreshold(0.5f)
            .setMaxResults(5)
            .build()

        return ObjectDetector.createFromFileAndOptions(
            this,
            "efficientdet-lite0.tflite",
            options
        )
    }
    // Present the result after detecting
    private fun runObjectDetection(bitmap: Bitmap) {
        val tensorImage = TensorImage.fromBitmap(bitmap)
        val results = detector.detect(tensorImage)

        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)

        val boxPaint = Paint().apply {
            style = Paint.Style.STROKE
            color = Color.RED
            strokeWidth = 6f
        }

        val textPaint = Paint().apply {
            color = Color.RED
            textSize = 48f
        }

        val labels = mutableListOf<String>()

        for (detection in results) {
            val category = detection.categories.firstOrNull() ?: continue
            val label = category.label
            val score = category.score
            val box = detection.boundingBox

            labels.add("$label (${String.format("%.2f", score)})")
            canvas.drawRect(box, boxPaint)
            canvas.drawText("$label ${"%.2f".format(score)}", box.left, box.top - 10, textPaint)
        }

        imageView.setImageBitmap(mutableBitmap)

        textView.text = if (labels.isEmpty()) "탐지된 객체 없음"
        else labels.joinToString(", ")
    }
}