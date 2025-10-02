package com.surendramaran.yolov8tflite

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.PorterDuff
import android.media.Image
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.surendramaran.yolov8tflite.Constants.LABELS_PATH
import com.surendramaran.yolov8tflite.Constants.MODEL_PATH
import yolov8tflite.R
import yolov8tflite.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeech.OnInitListener
import android.util.DisplayMetrics
import android.view.View
import android.view.WindowManager
import android.view.animation.AlphaAnimation
import androidx.camera.core.ImageProxy
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.async
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.consumeAsFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
import android.graphics.RectF
import kotlin.math.max
import kotlin.math.min
private const val DEPTH_SCALE_FACTOR = 0.0025f

class MainActivity : AppCompatActivity(), Detector.DetectorListener {
    private lateinit var binding: ActivityMainBinding
    private val isFrontCamera = false

    private var ttsReady = false
    private var detectorReady = false
    private var depthReady = false
    private var cameraReady = false

    private val DEPTH_MODEL_PATH = "Midas-V2_w8a8.tflite"

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var detector: Detector? = null
    private var depthEstimator: DepthEstimator? = null
    private var lastSpokenGuidance: String? = null

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var detectionExecutor: ExecutorService
    private lateinit var depthExecutor: ExecutorService
    private lateinit var tts: TextToSpeech

    private var bitmapBuffer: Bitmap? = null
    private var depthMap: Array<FloatArray>? = null
    private var lastWallRegion: RectF? = null
    private var lastWallSpokenTime: Long = 0L
    private val WALL_SPEECH_COOLDOWN_MS = 300L // Increased to 1 second for more stability
    // ==== Wall detection tunables (adjust here) ====
    private val WALL_DEPTH_VAR_THRESHOLD = 0.04f // Slightly more permissive for better detection
    private val WALL_ASPECT_RATIO_MIN = 1.0f     // More permissive: allow wider regions
    private val WALL_IOU_SUPPRESS_THRESHOLD = 0.10f // permissive: only suppress if large overlap with YOLO
    private val WALL_SUB_BANDS = 5              // finer vertical bands inside each grid cell (3-6)
    private val GRID_ROWS = 3
    private val GRID_COLS = 3
    // Distance filtering for wall detection
    private val WALL_MIN_DISTANCE_METERS = 0.3f  // Allow closer walls (was 0.8f)
    private val WALL_MAX_DISTANCE_METERS = 4.0f  // Allow farther walls (was 3.0f)
    private val WALL_FLOOR_EXCLUDE_HEIGHT = 0.8f // Exclude bottom 20% of frame (was 30%)
    private val WALL_MIN_SCORE_THRESHOLD = 0.8f  // Lower threshold for more sensitive detection (was 1.2f)
    // Wall detection improvements
    private val WALL_OPTIMAL_DISTANCE_MIN = 0.5f // Allow closer optimal range (was 1.0f)
    private val WALL_OPTIMAL_DISTANCE_MAX = 3.0f // Extend optimal range (was 2.0f)
    private val WALL_WARNING_DISTANCE_THRESHOLD = 1.5f // Only warn about walls within 1.5 meters
    private var wallDetected: Boolean = false
    private var lastWallVar: Float = 0f
    private var lastWallAspect: Float = 0f
    private var lastWallMean: Float = 0f
    private var lastWallScore: Float = 0f
    private var lastWallMeters: Float? = null
    
    // Wall detection smoothing and stability
    private var wallStateHistory = mutableListOf<Boolean>() // Track recent wall states
    private var wallDistanceHistory = mutableListOf<Float>() // Track recent wall distances
    private val WALL_STATE_HISTORY_SIZE = 3 // Number of frames to consider for smoothing (reduced for responsiveness)
    private val WALL_DISTANCE_HISTORY_SIZE = 1 // Number of distance measurements to average (reduced for responsiveness)

    // System failure detection
    private var systemFailureCounter = 0
    private var lastSystemFailureWarning = 0L
    private val SYSTEM_FAILURE_THRESHOLD = 30 // frames (about 10 seconds at 3fps)
    private val SYSTEM_FAILURE_WARNING_COOLDOWN_MS = 10000L // 10 seconds

    // Darkness detection
    private var darknessCounter = 0
    private var lastDarknessWarning = 0L
    private val DARKNESS_THRESHOLD = 5 // frames before warning
    private val DARKNESS_WARNING_COOLDOWN_MS = 5000L // 5 seconds
    private val DARKNESS_BRIGHTNESS_THRESHOLD = 30.0f // Average brightness below this triggers warning (0-255 scale)

    private var frameStep = 0;
    private val totalPipelineSteps = 3;

    private lateinit var loadingOverlay: View

    private var depthFrameCounter = 0
    private val DEPTH_SKIP_INTERVAL = 1 // Safe default, adjust as needed

    // HUD/debug metrics
    private var lastDetectionInferenceTime: Long = 0L
    private var lastDepthInferenceTime: Long = 0L
    private var lastLagMs: Long = 0L
    private var frameTimestampMs: Long = 0L
    private var depthSourceTimestampMs: Long = 0L
    private var lastDetectionFrameTimestampMs: Long = 0L
    private var lastBrightness: Float = 0f

    

    // Data class for region analysis
    private data class RegionAnalysis(
        val leftOccupied: Boolean,
        val centerOccupied: Boolean,
        val rightOccupied: Boolean,
        val aboveOccupied: Boolean,
        val belowOccupied: Boolean
    )
    private fun getDistanceDescription(meters: Float): String {
        return when {
            meters >= 2.5f -> "more than 2 meters ahead"
            else -> String.format("%.1f meters ahead", meters)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        loadingOverlay = binding.loadingOverlay

        cameraExecutor = Executors.newSingleThreadExecutor()
        detectionExecutor = Executors.newSingleThreadExecutor()
        depthExecutor = Executors.newSingleThreadExecutor()

        cameraExecutor.execute {
            // Initialize TTS
            tts = TextToSpeech(this, OnInitListener { status ->
                if (status == TextToSpeech.SUCCESS) {
                    tts.language = Locale.US // Set the language for TTS
                    tts.setOnUtteranceProgressListener(object : android.speech.tts.UtteranceProgressListener() {
                        override fun onStart(utteranceId: String?) {
                            isSpeaking = true
                        }
                        override fun onDone(utteranceId: String?) {
                            isSpeaking = false
                            lastSpokenTime = System.currentTimeMillis()
                            runOnUiThread {
                                speechQueue.removeFirstOrNull()?.let { next ->
                                    speakGuidance(next)
                                }
                            }
                        }
                        override fun onError(utteranceId: String?) {
                            isSpeaking = false
                        }
                    })
                    ttsReady = true
                    checkAllReady()
                    // Flush any queued speech
                    runOnUiThread {
                        speechQueue.removeFirstOrNull()?.let { next ->
                            speakGuidance(next)
                        }
                    }
                }
            })

            detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this) {
                toast(it)
            }
            detectorReady = true
            checkAllReady()

            depthEstimator = DepthEstimator(baseContext, DEPTH_MODEL_PATH)
            depthReady = true
            checkAllReady()
        }

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        val metrics = DisplayMetrics().also { binding.viewFinder.display.getRealMetrics(it) }
        val width = metrics.widthPixels
        val height = metrics.heightPixels

        if (bitmapBuffer == null || bitmapBuffer?.width != width || bitmapBuffer?.height != height) {
            bitmapBuffer = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        }

        val rotation = binding.viewFinder.display.rotation

        val cameraSelector = CameraSelector
            .Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        val screenWidth = binding.viewFinder.width
        val screenHeight = binding.viewFinder.height

        preview = Preview.Builder()
            .setTargetResolution(Size(width, height))
//            .setTargetResolution(analysisSize)  // Based on the throttle
            .setTargetRotation(rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetResolution(Size(screenWidth, screenHeight))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy ->
            // Record a frame timestamp in ms for consistent lag computation
            frameTimestampMs = System.currentTimeMillis()
            // 1) Ensure buffer is initialized and matches frame size
            if (bitmapBuffer == null ||
                bitmapBuffer!!.width  != imageProxy.width ||
                bitmapBuffer!!.height != imageProxy.height) {
                bitmapBuffer = Bitmap.createBitmap(
                    imageProxy.width,
                    imageProxy.height,
                    Bitmap.Config.ARGB_8888
                )
            }

            val bmp = bitmapBuffer!!
//            if (bmp.width  != imageProxy.width ||
//                bmp.height != imageProxy.height) {
//                // Re-create if resolution changed
//                bitmapBuffer = Bitmap.createBitmap(
//                    imageProxy.width,
//                    imageProxy.height,
//                    Bitmap.Config.ARGB_8888
//                )
//            }

            // 2) Copy raw camera bytes into the reused bitmap
            imageProxy.planes[0].buffer.rewind()  // reset position
            bmp.copyPixelsFromBuffer(imageProxy.planes[0].buffer)

            // 3) Rotate/mirror into a new Bitmap only if needed by your model
            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                if (isFrontCamera) postScale(-1f, 1f, bmp.width / 2f, bmp.height / 2f)
            }
            val rotatedBitmap = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)

            // detector?.detect(rotatedBitmap, screenWidth, screenHeight)
//            detectionExecutor.execute {
//                detector?.detect(rotatedBitmap, screenWidth, screenHeight)
//            }
//
//            depthExecutor.execute {
//                val depthResult = depthEstimator?.estimateDepth(rotatedBitmap)
//                runOnUiThread {
//                    depthResult?.let { result ->
//                        result.depthBitmap?.let { bitmap ->
//                            binding.depthView.setImageBitmap(bitmap)
//                            binding.overlay.setDepthMap(result.rawDepthArray)
//                            depthMap = result.rawDepthArray
//                        }
//                    }
//                }
//            }

            // Implementing a pipeline
            when (frameStep % totalPipelineSteps) {
                0 -> {
                    detectionExecutor.execute {
                        lastDetectionFrameTimestampMs = frameTimestampMs
                        detector?.detect(rotatedBitmap, screenWidth, screenHeight);
                    }
                    // Also check darkness on every frame for safety
                    runOnUiThread {
                        checkDarkness(rotatedBitmap)
                    }
                }
                1 -> {
                    if (depthFrameCounter % DEPTH_SKIP_INTERVAL == 0) {
                        depthExecutor.execute {
                            val startTime = System.currentTimeMillis()
                            val sourceTs = frameTimestampMs
                            val depthResult = depthEstimator?.estimateDepth(rotatedBitmap)
                            val elapsed = System.currentTimeMillis() - startTime
                            val depthTimestamp = System.currentTimeMillis()
                            val lag = depthTimestamp - frameTimestampMs
                            lastDepthInferenceTime = elapsed
                            lastLagMs = lag
                            runOnUiThread {
                                val raw = depthResult?.rawDepthArray
                                if (raw != null) {
                                    binding.overlay.setDepthMap(raw)
                                    depthMap = raw
                                    depthSourceTimestampMs = sourceTs
                                    systemFailureCounter = 0 // Reset system failure counter on successful depth update
                                    // Wall detection is now called directly in navigation functions for real-time updates
                                }
                                updateHud()
                            }
                            Log.d("DEPTH_DEBUG", "Depth inference: ${elapsed}ms, lag from frame: ${lag}ms")
                        }
                    }
                    depthFrameCounter++
                }
                2 -> {
                    runOnUiThread {
                        lastSpokenGuidance?.let { guidance ->
                            speakGuidance(guidance)
                        }
                    }
                }
            }
            frameStep++
            imageProxy.close()
        }

        cameraProvider.unbindAll()

        try {
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            preview?.surfaceProvider = binding.viewFinder.surfaceProvider
            cameraReady = true
            checkAllReady()
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) {
        if (it[Manifest.permission.CAMERA] == true) {
            startCamera()
        }
    }

    private fun speakGuidance(guidance: String, objectId: String? = null) {
        val now = System.currentTimeMillis()
        // Per-object cooldown
        if (objectId != null) {
            val lastObjectSpoken = spokenObjectTimestamps[objectId] ?: 0
            if (now - lastObjectSpoken < OBJECT_ALERT_COOLDOWN_MS) return
            spokenObjectTimestamps[objectId] = now
        }
        // Global cooldown and queue
        if (!ttsReady) {
            // TTS not ready, queue the guidance
            if (!speechQueue.contains(guidance)) {
                speechQueue.addLast(guidance)
            }
            return
        }
        if (isSpeaking || now - lastSpokenTime < SPEECH_COOLDOWN_MS) {
            if (!speechQueue.contains(guidance)) {
                speechQueue.addLast(guidance)
            }
            return
        }
        isSpeaking = true
        lastSpokenTime = now
        tts.speak(guidance, TextToSpeech.QUEUE_FLUSH, null, "GUIDANCE")
    }

    private fun toast(message: String) {
        runOnUiThread {
            Toast.makeText(baseContext, message, Toast.LENGTH_LONG).show()
        }
    }

    private fun checkAllReady() {
        if (ttsReady && detectorReady && depthReady && cameraReady) {
            runOnUiThread {
                loadingOverlay.visibility = View.GONE // Hide loading spinner/overlay
                // You can add other actions here if needed, e.g.:
                // toast("System ready!")
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        tts.shutdown()
        detector?.close()
        cameraExecutor.shutdown()
        detectionExecutor.shutdown()
        depthExecutor.shutdown()
    }

    override fun onResume() {
        super.onResume()
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    companion object {
        private const val TAG = "Camera"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = mutableListOf(
            Manifest.permission.CAMERA
        ).toTypedArray()
    }

    override fun onEmptyDetect() {
        runOnUiThread {
            binding.overlay.clear()
            emptyDetectionsStreak++
            
            // Check for system failure (both detector and depth not working)
            checkSystemFailure()
            
            if (emptyDetectionsStreak >= REQUIRED_EMPTY_STREAK) {
                // Run wall detection first to get current state
                // (updateWallDetection now calls updateHud() internally for sync)
                updateWallDetection(emptyList())
                
                // Generate path clear guidance when no objects are detected
                val pathClearGuidance = generatePathClearGuidance()
                if (pathClearGuidance != null && pathClearGuidance != lastSpokenGuidance) {
                    lastSpokenGuidance = pathClearGuidance
                    speakGuidance(pathClearGuidance, "path_clear")
                }
            }
        }
    }

    // Add these fields to MainActivity (top of class, after other vars)
    private val objectHistory = mutableMapOf<String, ObjectTrack>()
    private val OBJECT_PERSISTENCE_FRAMES = 10
    private val MOVEMENT_THRESHOLD = 0.04f // normalized movement threshold
    private val DEPTH_THRESHOLD = 0.5f // meters


    data class ObjectTrack(
        val id: String,
        var lastFrame: Int,
        var lastX: Float,
        var lastY: Float,
        var lastDepth: Float?
    )

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        runOnUiThread {
            // Reset empty-frame streak when we have detections
            emptyDetectionsStreak = 0
            systemFailureCounter = 0 // Reset system failure counter on successful detection
            lastDetectionInferenceTime = inferenceTime
            // HUD will be updated after wall detection for proper sync
            // Filter boxes by distance (0.5m to 5.0m)
            val filteredBoxes = boundingBoxes
            binding.overlay.apply {
                setResults(filteredBoxes)
                invalidate()
            }

            if (filteredBoxes.isNotEmpty()) {
                val regionAnalysis = analyzeRegions(filteredBoxes)
                val guidance = generateNavigationGuidance(
                    regionAnalysis.leftOccupied,
                    regionAnalysis.centerOccupied,
                    regionAnalysis.rightOccupied,
                    regionAnalysis.aboveOccupied,
                    regionAnalysis.belowOccupied,
                    filteredBoxes
                )

                // Re-run wall detection after new detections to allow YOLO-overlap suppression
                updateWallDetection(filteredBoxes)

                // Object persistence check
                val shouldSpeak = filteredBoxes.any { box ->
                    val objectId = box.clsName // Use class+region as ID, or add more info if needed
                    val centerX = (box.x1 + box.x2) / 2f
                    val centerY = (box.y1 + box.y2) / 2f
                    val depth = depthMap?.let { depthArray ->
                        val x = (centerX * (depthArray[0].size - 1)).toInt()
                        val y = (centerY * (depthArray.size - 1)).toInt()
                        if (y in depthArray.indices && x in depthArray[0].indices) {
                            RawDepth(depthArray[y][x]).toMeters()
                        } else null
                    }

                    val track = objectHistory[objectId]
                    val moved = track == null ||
                        (Math.abs(centerX - track.lastX) > MOVEMENT_THRESHOLD) ||
                        (Math.abs(centerY - track.lastY) > MOVEMENT_THRESHOLD) ||
                        (depth != null && track.lastDepth != null && Math.abs(depth - track.lastDepth!!) > DEPTH_THRESHOLD) ||
                        (frameStep - track.lastFrame > OBJECT_PERSISTENCE_FRAMES)

                    // Update history
                    objectHistory[objectId] = ObjectTrack(
                        id = objectId,
                        lastFrame = frameStep,
                        lastX = centerX,
                        lastY = centerY,
                        lastDepth = depth
                    )

                    moved
                }

                // Only speak if guidance is not null, different, and object is new/moved
                if (guidance != null && guidance != lastSpokenGuidance && shouldSpeak) {
                    lastSpokenGuidance = guidance
                    val primaryObjectId = filteredBoxes.firstOrNull()?.clsName ?: "general"
                    speakGuidance(guidance, primaryObjectId)
                }
            } else {
                // Run wall detection first to get current state
                updateWallDetection(emptyList())
            }
        }
    }

    private fun analyzeRegions(boundingBoxes: List<BoundingBox>): RegionAnalysis {
        var leftOccupied = false
        var centerOccupied = false
        var rightOccupied = false
        var aboveOccupied = false
        var belowOccupied = false

        boundingBoxes.forEach { box ->
            when {
                box.clsName.endsWith("-left") -> leftOccupied = true
                box.clsName.endsWith("-center") -> centerOccupied = true
                box.clsName.endsWith("-right") -> rightOccupied = true
                box.clsName.endsWith("-above") -> aboveOccupied = true
                box.clsName.endsWith("-below") -> belowOccupied = true
            }
        }

        return RegionAnalysis(leftOccupied, centerOccupied, rightOccupied, aboveOccupied, belowOccupied)
    }

    private fun generateNavigationGuidance(
        leftOccupied: Boolean,
        centerOccupied: Boolean,
        rightOccupied: Boolean,
        aboveOccupied: Boolean,
        belowOccupied: Boolean,
        boxes: List<BoundingBox>
    ): String? {
        // Centralized, optimized grouping by depth (single pass, sorted)
        fun clusterByDepthFast(
            boxes: List<BoundingBox>,
            depthMap: Array<FloatArray>?,
            threshold: Float
        ): List<List<BoundingBox>> {
            if (depthMap == null) return listOf(boxes)
            val boxesWithDepth = boxes.mapNotNull { box ->
                val depth = getBoxDepth(box, depthMap)
                if (depth != null) Pair(box, depth) else null
            }.sortedBy { it.second }
            val clusters = mutableListOf<MutableList<BoundingBox>>()
            var currentCluster = mutableListOf<BoundingBox>()
            var lastDepth: Float? = null
            for ((box, depth) in boxesWithDepth) {
                if (lastDepth == null || kotlin.math.abs(depth - lastDepth) < threshold) {
                    currentCluster.add(box)
                } else {
                    clusters.add(currentCluster)
                    currentCluster = mutableListOf(box)
                }
                lastDepth = depth
            }
            if (currentCluster.isNotEmpty()) clusters.add(currentCluster)
            return clusters
        }

        // Group people by depth
        val personBoxes = boxes.filter { it.clsName.startsWith("person") }
        val personGroups = clusterByDepthFast(personBoxes, depthMap, 2.0f) // 1.0m threshold

        if (personGroups.isNotEmpty()) {
            // Find the largest group and its region
            val largestGroup = personGroups.maxByOrNull { it.size }
            val groupRegion = largestGroup?.let { group ->
                val regionCounts = group.groupingBy { it.clsName.substringAfter("-") }.eachCount()
                regionCounts.maxByOrNull { it.value }?.key ?: "ahead"
            }
            val groupDistance = largestGroup?.let { group ->
                group.mapNotNull { box ->
                    getBoxDepth(box, depthMap)
                }.minOrNull()
            }

            if (largestGroup != null && largestGroup.size > 1 && groupDistance != null && groupDistance in 0.5f..5.0f) {
                val distanceDescription = String.format("%.1f meters", groupDistance)
                return "people $groupRegion $distanceDescription ahead"
            }
        }

        // Single object guidance (most prominent)
        val primaryObject = boxes.maxByOrNull { it.cnf * it.w * it.h }
        val objectName = primaryObject?.clsName?.substringBefore("-") ?: "object"
        val region = primaryObject?.clsName?.substringAfter("-") ?: "ahead"
        val distance = primaryObject?.let { box ->
            getBoxDepth(box, depthMap)
        }
        if (distance != null && (distance < 0.5f || distance > 5.0f)) {
            return null
        }
        val distanceDescription = distance?.let { String.format("%.1f meters", it) } ?: ""

        // Priority checks (safety first)
        when {
            belowOccupied -> return "$objectName below, stop immediately"
            aboveOccupied && !(leftOccupied || centerOccupied || rightOccupied) ->
                return "$objectName above, lower your head"
        }

        // After the person group logic, always check for the closest object:
        val closestBox = boxes.minByOrNull { getBoxDepth(it, depthMap) ?: Float.MAX_VALUE }
        val closestDistance = closestBox?.let { getBoxDepth(it, depthMap) }
        if (closestBox != null && closestDistance != null && closestDistance > 0f && closestDistance < 0.5f) {
            val objectName = closestBox.clsName.substringBefore("-")
            return "$objectName very 4close, stop!"
        }

        if (leftOccupied && !rightOccupied) {
            return if (!centerOccupied) "$objectName left $distanceDescription ahead, move right"
            else "$objectName left and center $distanceDescription ahead, move further right"
        } else if (rightOccupied && !leftOccupied) {
            return if (!centerOccupied) "$objectName right $distanceDescription ahead, move left"
            else "$objectName right and center $distanceDescription ahead, move further left"
        } else if (centerOccupied) {
            return when {
                !leftOccupied && !rightOccupied -> "$objectName center $distanceDescription ahead, move left or right"
                !leftOccupied -> "$objectName center $distanceDescription ahead, move left"
                !rightOccupied -> "$objectName center $distanceDescription ahead, move right"
                else -> "$objectName $distanceDescription ahead blocking path, stop"
            }
        } else if (leftOccupied && rightOccupied && !centerOccupied) {
            // Check if center path is actually clear using depth data
            val centerPathClear = isCenterPathClear()
            // Also check for close walls
            val wallClose = wallDetected && lastWallMeters != null && lastWallMeters!! < WALL_WARNING_DISTANCE_THRESHOLD
            return if (centerPathClear && !wallClose) {
                "Objects on both sides $distanceDescription ahead, center path is clear, proceed forward"
            } else if (wallClose) {
                val wallDistancePart = if (lastWallMeters != null) String.format("%.1f meters", lastWallMeters) else ""
                if (wallDistancePart.isNotEmpty()) {
                    "Wall ahead $wallDistancePart, be careful, feel what's in front of you and stop"
                } else {
                    "Wall ahead, be careful, feel what's in front of you and stop"
                }
            } else {
                "Objects on both sides $distanceDescription ahead, proceed carefully forward"
            }
        } else if (!leftOccupied && !centerOccupied && !rightOccupied && !aboveOccupied && !belowOccupied) {
            // No objects detected in any region
            return null
        } else {
            // Default case when no specific guidance applies
            // Check for close walls before giving any guidance
            val wallClose = wallDetected && lastWallMeters != null && lastWallMeters!! < WALL_WARNING_DISTANCE_THRESHOLD
            if (wallClose) {
                val distancePart = if (lastWallMeters != null) String.format("%.1f meters", lastWallMeters) else ""
                return if (distancePart.isNotEmpty()) {
                    "Wall ahead $distancePart, be careful, feel what's in front of you and stop"
                } else {
                    "Wall ahead, be careful, feel what's in front of you and stop"
                }
            }
            
            // No specific guidance needed
            return null
        }
    }

    // Helper function for getting box depth (already present in your code)
    private fun getBoxDepth(box: BoundingBox, depthMap: Array<FloatArray>?): Float? {
        return depthMap?.let { depthArray ->
            val centerX = ((box.x1 + box.x2) / 2 * (depthArray[0].size - 1)).toInt()
            val centerY = ((box.y1 + box.y2) / 2 * (depthArray.size - 1)).toInt()
            val medianRaw = getMedianDepthPatch(depthArray, centerX, centerY, patchSize = 2)
            medianRaw?.let { RawDepth(it).toMeters() }
        }
    }

    private val spokenObjectTimestamps = mutableMapOf<String, Long>()
    private val OBJECT_ALERT_COOLDOWN_MS = 1000L // 5 seconds per object
    private var lastSpokenTime: Long = 0
    private val SPEECH_COOLDOWN_MS = 1000L // 3.5 seconds global debounce

    private val speechQueue: ArrayDeque<String> = ArrayDeque()
    private var isSpeaking: Boolean = false

    // Add to MainActivity.kt (inside the class)
    private fun getMedianDepthPatch(
        depthArray: Array<FloatArray>,
        centerX: Int,
        centerY: Int,
        patchSize: Int = 2
    ): Float? {
        val depths = mutableListOf<Float>()
        for (dy in -patchSize..patchSize) {
            for (dx in -patchSize..patchSize) {
                val px = centerX + dx
                val py = centerY + dy
                if (py in depthArray.indices && px in depthArray[0].indices) {
                    depths.add(depthArray[py][px])
                }
            }
        }
        if (depths.isEmpty()) return null
        return depths.sorted()[depths.size / 2]
    }



    @JvmInline
    private value class RawDepth(val value: Float)

    private inline fun RawDepth.toMeters(): Float = 1.0f / (value * DEPTH_SCALE_FACTOR)

    // Debounce for empty detections to avoid saying "Path clear" between detection frames
    private var emptyDetectionsStreak = 0
    private val REQUIRED_EMPTY_STREAK = 2 // Reduced for faster response

    private fun generatePathClearGuidance(): String? {
        val depth = depthMap ?: return null
        
        // Check if there's a close wall that would block the path
        val wallClose = wallDetected && lastWallMeters != null && lastWallMeters!! < WALL_WARNING_DISTANCE_THRESHOLD
        if (wallClose) {
            // Don't announce path clear if there's a close wall
            return null
        }
        
        // Check if the forward corridor is clear
        val corridorClear = isForwardCorridorClear(depth)
        if (corridorClear) {
            return "Path clear, proceed forward"
        }
        
        return null
    }

    private fun checkSystemFailure() {
        // Check if both detector and depth estimation are failing
        val depthAge = if (depthSourceTimestampMs > 0L) System.currentTimeMillis() - depthSourceTimestampMs else -1L
        val depthFailing = depthMap == null || depthAge > 5000L // Depth data older than 5 seconds
        val detectorFailing = emptyDetectionsStreak > 10 // No detections for extended period
        
        if (depthFailing && detectorFailing) {
            systemFailureCounter++
            
            // Warn user if system has been failing for too long
            val now = System.currentTimeMillis()
            if (systemFailureCounter >= SYSTEM_FAILURE_THRESHOLD && now - lastSystemFailureWarning > SYSTEM_FAILURE_WARNING_COOLDOWN_MS) {
                val warningMsg = "Warning: Navigation system may not be working properly. Please be extra careful and consider stopping."
                speakGuidance(warningMsg, "system_failure")
                lastSystemFailureWarning = now
                Log.w("SYSTEM_FAILURE", "System failure detected: depth failing=$depthFailing, detector failing=$detectorFailing, counter=$systemFailureCounter")
            }
        } else {
            // Reset counter if either system is working
            if (systemFailureCounter > 0) {
                systemFailureCounter = max(0, systemFailureCounter - 2) // Slow recovery
            }
        }
    }

    private fun checkDarkness(bitmap: Bitmap) {
        val brightness = calculateAverageBrightness(bitmap)
        lastBrightness = brightness // Store for HUD display
        
        if (brightness < DARKNESS_BRIGHTNESS_THRESHOLD) {
            darknessCounter++
            
            // Warn user if it's been dark for too many frames
            val now = System.currentTimeMillis()
            if (darknessCounter >= DARKNESS_THRESHOLD && now - lastDarknessWarning > DARKNESS_WARNING_COOLDOWN_MS) {
                val warningMsg = "Warning: Environment is too dark for safe navigation. Please stop and find better lighting or assistance."
                speakGuidance(warningMsg, "darkness")
                lastDarknessWarning = now
                Log.w("DARKNESS_WARNING", "Dark environment detected: brightness=$brightness, threshold=$DARKNESS_BRIGHTNESS_THRESHOLD")
            }
        } else {
            // Reset counter when brightness is adequate
            darknessCounter = 0
        }
    }

    private fun calculateAverageBrightness(bitmap: Bitmap): Float {
        val width = bitmap.width
        val height = bitmap.height
        var totalBrightness = 0L
        var pixelCount = 0
        
        // Sample every 4th pixel for performance (still gives accurate average)
        val step = 4
        for (y in 0 until height step step) {
            for (x in 0 until width step step) {
                val pixel = bitmap.getPixel(x, y)
                
                // Calculate luminance using standard weights
                val red = (pixel shr 16) and 0xFF
                val green = (pixel shr 8) and 0xFF
                val blue = pixel and 0xFF
                
                // Standard luminance formula: 0.299*R + 0.587*G + 0.114*B
                val brightness = (0.299 * red + 0.587 * green + 0.114 * blue).toInt()
                totalBrightness += brightness
                pixelCount++
            }
        }
        
        return if (pixelCount > 0) totalBrightness.toFloat() / pixelCount else 0f
    }

    // Depth-based forward corridor check (bottom-center region clear and reasonably far)
    private fun isForwardCorridorClear(depth: Array<FloatArray>): Boolean {
        // If a wall is detected but it's far away (beyond warning threshold), consider corridor clear
        if (wallDetected && lastWallMeters != null && lastWallMeters!! >= WALL_WARNING_DISTANCE_THRESHOLD) {
            return true // Wall is far, corridor is clear
        }
        
        // Sample a rectangle in the lower middle of the frame
        val xStartN = 0.33f
        val xEndN = 0.66f
        val yStartN = 0.60f
        val yEndN = 0.95f

        val xStart = (xStartN * (depth[0].size - 1)).toInt()
        val xEnd = ((xEndN * (depth[0].size - 1)).toInt()).coerceAtLeast(xStart + 1)
        val yStart = (yStartN * (depth.size - 1)).toInt()
        val yEnd = ((yEndN * (depth.size - 1)).toInt()).coerceAtLeast(yStart + 1)

        var minMeters = Float.MAX_VALUE
        var samples = 0
        val stepY = ((yEnd - yStart) / 8).coerceAtLeast(1)
        val stepX = ((xEnd - xStart) / 8).coerceAtLeast(1)

        var y = yStart
        while (y <= yEnd) {
            var x = xStart
            while (x <= xEnd) {
                val raw = depth[y][x]
                if (raw > 0f) {
                    val m = RawDepth(raw).toMeters()
                    if (m < minMeters) minMeters = m
                    samples++
                }
                x += stepX
            }
            y += stepY
        }
        if (samples == 0) return false
        // Consider corridor clear if the closest sample is at least 1.2 m away
        return minMeters >= 1.2f
    }

    private fun updateHud() {
        val det = lastDetectionInferenceTime
        val dep = lastDepthInferenceTime
        val lag = lastLagMs
        val depthAgeMs = if (depthSourceTimestampMs > 0L) System.currentTimeMillis() - depthSourceTimestampMs else -1L

        fun speedLabel(ms: Long, fast: Long, slow: Long): String {
            return when {
                ms <= fast -> "fast"
                ms <= slow -> "ok"
                else -> "slow"
            }
        }

        val detLabel = speedLabel(det, fast = 30, slow = 60)
        val depLabel = speedLabel(dep, fast = 40, slow = 100)
        val lagLabel = speedLabel(lag, fast = 80, slow = 150)
        val ageLabel = if (depthAgeMs >= 0) speedLabel(depthAgeMs, fast = 80, slow = 160) else "n/a"

        val statusPriority = mapOf("fast" to 0, "ok" to 1, "slow" to 2)
        val worstLabel = listOf(detLabel, depLabel, lagLabel, ageLabel).maxByOrNull { statusPriority[it] ?: 0 } ?: "ok"
        val color = when (worstLabel) {
            "fast" -> Color.GREEN
            "ok" -> Color.YELLOW
            else -> Color.RED
        }

        binding.inferenceTime.setTextColor(color)
        val ageText = if (depthAgeMs >= 0) "\nDepthAge: ${depthAgeMs}ms (${ageLabel})" else ""
        binding.inferenceTime.isSingleLine = false
        binding.inferenceTime.maxLines = 6
        binding.inferenceTime.textAlignment = View.TEXT_ALIGNMENT_VIEW_START
        val wallMetersText = lastWallMeters?.let { String.format("%.1f m", it) } ?: "n/a"
        val wallText = if (wallDetected) {
            "WALL yes score=" + String.format("%.2f", lastWallScore) +
            " var=" + String.format("%.4f", lastWallVar) +
            " asp=" + String.format("%.2f", lastWallAspect) +
            " mean=" + String.format("%.2f", lastWallMean) +
            " dist=" + wallMetersText
        } else {
            "WALL no"
        }
        val brightnessStatus = if (lastBrightness < DARKNESS_BRIGHTNESS_THRESHOLD) "DARK" else "OK"
        val brightnessText = "\nBrightness: ${String.format("%.1f", lastBrightness)} ($brightnessStatus)"
        binding.inferenceTime.text = "Det: ${det}ms (${detLabel})\nDepth: ${dep}ms (${depLabel})\nLag: ${lag}ms (${lagLabel})${ageText}\n" + wallText + brightnessText
    }

    // -------------------- WALL DETECTION --------------------
    private fun updateWallDetection(currentDetections: List<BoundingBox>? = null) {
        val depth = depthMap ?: run {
            binding.overlay.setWallRegion(null)
            return
        }

        // Normalize depth to 0..1 (per-frame min-max)
        var minV = Float.MAX_VALUE
        var maxV = -Float.MAX_VALUE
        for (row in depth) {
            for (v in row) {
                if (v < minV) minV = v
                if (v > maxV) maxV = v
            }
        }
        val range = max(1e-6f, maxV - minV)

        // Prepare YOLO boxes for overlap suppression
        val yoloBoxes = (currentDetections ?: listOf()).map { box ->
            RectF(box.x1, box.y1, box.x2, box.y2)
        }

        // Scan grid (and vertical bands inside each cell to allow tall regions)
        var bestRegion: RectF? = null
        var bestScore = Float.NEGATIVE_INFINITY
        val candidateRegions = mutableListOf<Pair<RectF, Float>>() // Store all good candidates

        for (r in 0 until GRID_ROWS) {
            for (c in 0 until GRID_COLS) {
                val yStartN = r / GRID_ROWS.toFloat()
                val yEndN = (r + 1) / GRID_ROWS.toFloat()
                val xStartN = c / GRID_COLS.toFloat()
                val xEndN = (c + 1) / GRID_COLS.toFloat()

                // Skip floor regions (bottom portion of frame)
                if (yStartN > WALL_FLOOR_EXCLUDE_HEIGHT) continue

                // Subdivide horizontally into vertical bands to capture tall narrow walls
                for (b in 0 until WALL_SUB_BANDS) {
                    val bxStartN = xStartN + (b / WALL_SUB_BANDS.toFloat()) * (xEndN - xStartN)
                    val bxEndN = xStartN + ((b + 1) / WALL_SUB_BANDS.toFloat()) * (xEndN - xStartN)

                    val yStart = (yStartN * (depth.size - 1)).toInt()
                    val yEnd = ((yEndN * (depth.size - 1)).toInt()).coerceAtLeast(yStart + 1)
                    val xStart = (bxStartN * (depth[0].size - 1)).toInt()
                    val xEnd = ((bxEndN * (depth[0].size - 1)).toInt()).coerceAtLeast(xStart + 1)

                    var sum = 0.0
                    var sumSq = 0.0
                    var count = 0
                    var minDepthRaw = Float.MAX_VALUE
                    var maxDepthRaw = -Float.MAX_VALUE
                    
                    for (y in yStart..yEnd) {
                        val row = depth[y]
                        for (x in xStart..xEnd) {
                            val rawDepth = row[x]
                            if (rawDepth > 0f) {
                                minDepthRaw = min(minDepthRaw, rawDepth)
                                maxDepthRaw = max(maxDepthRaw, rawDepth)
                            }
                            val norm = ((rawDepth - minV) / range)
                            sum += norm
                            sumSq += norm * norm
                            count++
                        }
                    }
                    if (count == 0) continue
                    val mean = (sum / count).toFloat()
                    val varVal = ((sumSq / count) - (mean * mean)).toFloat()

                    // Convert raw depth to meters for distance filtering
                    val minDepthMeters = if (minDepthRaw < Float.MAX_VALUE) RawDepth(minDepthRaw).toMeters() else null
                    val maxDepthMeters = if (maxDepthRaw > -Float.MAX_VALUE) RawDepth(maxDepthRaw).toMeters() else null
                    
                    // Skip regions that are too close (likely floor) or too far (likely background)
                    if (minDepthMeters != null && minDepthMeters < WALL_MIN_DISTANCE_METERS) continue
                    if (maxDepthMeters != null && maxDepthMeters > WALL_MAX_DISTANCE_METERS) continue

                    val regionWidthN = (bxEndN - bxStartN)
                    val regionHeightN = (yEndN - yStartN)
                    val aspect = (regionHeightN / max(1e-6f, regionWidthN))

                    if (varVal > WALL_DEPTH_VAR_THRESHOLD) continue
                    if (aspect < WALL_ASPECT_RATIO_MIN) continue

                    val candidateRect = RectF(bxStartN, yStartN, bxEndN, yEndN)
                    val overlapsDetection = yoloBoxes.any { yb ->
                        intersectionOverUnion(candidateRect, yb) > WALL_IOU_SUPPRESS_THRESHOLD
                    }
                    if (overlapsDetection) continue

                    // Calculate distance-based scoring - prefer walls at optimal distances
                    val avgDepthMeters = if (minDepthMeters != null && maxDepthMeters != null) {
                        (minDepthMeters + maxDepthMeters) / 2f
                    } else null
                    
                    val distanceScore = if (avgDepthMeters != null) {
                        when {
                            avgDepthMeters in WALL_OPTIMAL_DISTANCE_MIN..WALL_OPTIMAL_DISTANCE_MAX -> 1.0f // Optimal range
                            avgDepthMeters < WALL_OPTIMAL_DISTANCE_MIN -> 0.7f // Close walls are still valid (was 0.3f)
                            avgDepthMeters > WALL_OPTIMAL_DISTANCE_MAX -> 0.5f // Too far (likely background)
                            else -> 0.0f
                        }
                    } else 0.0f
                    
                    // Check for depth gradient - walls should have consistent depth across the region
                    // Free hallway space often has depth gradients (closer floor, farther ceiling)
                    val depthGradientScore = if (minDepthMeters != null && maxDepthMeters != null) {
                        val depthRange = maxDepthMeters - minDepthMeters
                        when {
                            depthRange < 0.3f -> 1.0f // Very consistent depth (likely wall)
                            depthRange < 0.6f -> 0.7f // Moderately consistent
                            else -> 0.2f // Large depth variation (likely free space with floor/ceiling)
                        }
                    } else 0.0f
                    
                    // Combine flatness (low variance) with optimal distance and depth consistency
                    val flatnessScore = (1.0f - varVal * 10f).coerceAtLeast(0f)
                    val score = flatnessScore + distanceScore + depthGradientScore

                    if (score > bestScore) {
                        bestScore = score
                        bestRegion = candidateRect
                        lastWallVar = varVal
                        lastWallAspect = aspect
                        lastWallMean = mean
                    }
                    
                    // Store all good candidates for potential merging
                    if (score >= WALL_MIN_SCORE_THRESHOLD) {
                        candidateRegions.add(Pair(candidateRect, score))
                    }
                }
            }
        }

        // Try to merge adjacent wall regions for a more accurate representation
        val finalWallRegion = if (bestRegion != null && bestScore >= WALL_MIN_SCORE_THRESHOLD) {
            mergeAdjacentWallRegions(candidateRegions, bestRegion)
        } else null
        
        lastWallRegion = finalWallRegion
        binding.overlay.setWallRegion(finalWallRegion)

        // Apply smoothing to wall detection
        val currentWallDetected = finalWallRegion != null
        val currentDistance = if (finalWallRegion != null) approximateRegionDistanceMeters(finalWallRegion, depth) else null
        val (smoothedWallDetected, smoothedDistance) = smoothWallDetection(currentWallDetected, currentDistance)
        
        // Update debug state and logs FIRST to ensure consistency
        wallDetected = smoothedWallDetected
        lastWallScore = bestScore
        lastWallMeters = smoothedDistance
        
        // TTS if a smoothed wall is detected AND it's close enough to warn about
        val now = System.currentTimeMillis()
        if (wallDetected && lastWallMeters != null && lastWallMeters!! < WALL_WARNING_DISTANCE_THRESHOLD && now - lastWallSpokenTime > WALL_SPEECH_COOLDOWN_MS) {
            val distancePart = if (lastWallMeters!! in 0.5f..5.0f) String.format("%.1f meters", lastWallMeters!!) else ""
            val msg = if (distancePart.isNotEmpty()) {
                "Wall ahead $distancePart, be careful, feel what's in front of you and stop"
            } else {
                "Wall ahead, be careful, feel what's in front of you and stop"
            }
            speakGuidance(msg, "wall")
            lastWallSpokenTime = now
        }
        val consensusCount = wallStateHistory.count { it }
        Log.d("WALL_DEBUG", "raw=$currentWallDetected smoothed=$wallDetected history=${wallStateHistory} consensus=${consensusCount}/3 score=${String.format("%.3f", bestScore)} var=${String.format("%.5f", lastWallVar)} asp=${String.format("%.2f", lastWallAspect)} mean=${String.format("%.3f", lastWallMean)} meters=${lastWallMeters ?: -1f}")

        // Update on-screen wall debug text
        val wallMetersText = lastWallMeters?.let { String.format("%.1f m", it) } ?: "n/a"
        val debugText = if (wallDetected) {
            val isCloseEnough = lastWallMeters != null && lastWallMeters!! < WALL_WARNING_DISTANCE_THRESHOLD
            val warningStatus = if (isCloseEnough) "WARN" else "FAR"
            "WALL $warningStatus s=" + String.format("%.2f", lastWallScore) +
            " v=" + String.format("%.4f", lastWallVar) +
            " a=" + String.format("%.2f", lastWallAspect) +
            " m=" + String.format("%.2f", lastWallMean) +
            " d=" + wallMetersText
        } else {
            "WALL no"
        }
        binding.overlay.setWallDebugText(debugText)
        
        // Update HUD immediately to sync with wall detection state
        updateHud()
    }

    private fun intersectionOverUnion(a: RectF, b: RectF): Float {
        val ix1 = max(a.left, b.left)
        val iy1 = max(a.top, b.top)
        val ix2 = min(a.right, b.right)
        val iy2 = min(a.bottom, b.bottom)
        val interW = max(0f, ix2 - ix1)
        val interH = max(0f, iy2 - iy1)
        val inter = interW * interH
        val areaA = (a.right - a.left) * (a.bottom - a.top)
        val areaB = (b.right - b.left) * (b.bottom - b.top)
        val denom = max(1e-6f, areaA + areaB - inter)
        return inter / denom
    }

    private fun approximateRegionDistanceMeters(region: RectF, depth: Array<FloatArray>): Float? {
        // Sample center of region
        val cxN = (region.left + region.right) / 2f
        val cyN = (region.top + region.bottom) / 2f
        val x = (cxN * (depth[0].size - 1)).toInt()
        val y = (cyN * (depth.size - 1)).toInt()
        if (y !in depth.indices || x !in depth[0].indices) return null
        val raw = getMedianDepthPatch(depth, x, y, patchSize = 2) ?: return null
        return RawDepth(raw).toMeters()
    }

    private fun isWallBlockingForwardPath(wallRegion: RectF, depth: Array<FloatArray>): Boolean {
        // Check if the wall is in the center-forward path (not just on the sides)
        val centerPathLeft = 0.3f   // Left boundary of center path
        val centerPathRight = 0.7f  // Right boundary of center path
        val centerPathTop = 0.3f    // Top boundary of center path (avoid ceiling)
        val centerPathBottom = 0.8f // Bottom boundary of center path (avoid floor)
        
        // Check if wall region overlaps with the center-forward path
        val wallInCenterPath = wallRegion.left < centerPathRight && 
                              wallRegion.right > centerPathLeft &&
                              wallRegion.top < centerPathBottom && 
                              wallRegion.bottom > centerPathTop
        
        if (!wallInCenterPath) {
            // Wall is on the side, not blocking forward path
            return false
        }
        
        // Wall is in center path, check if it's close enough to be a concern
        val wallDistance = approximateRegionDistanceMeters(wallRegion, depth)
        return wallDistance != null && wallDistance < WALL_WARNING_DISTANCE_THRESHOLD
    }

    private fun isCenterPathClear(): Boolean {
        val depth = depthMap ?: return false
        
        // If a wall is detected and it's close (within warning threshold), path is NOT clear
        if (wallDetected && lastWallMeters != null && lastWallMeters!! < WALL_WARNING_DISTANCE_THRESHOLD) {
            return false // Wall is close, path is blocked
        }
        
        // If a wall is detected but it's far away (beyond warning threshold), consider path clear
        if (wallDetected && lastWallMeters != null && lastWallMeters!! >= WALL_WARNING_DISTANCE_THRESHOLD) {
            return true // Wall is far, path is clear
        }
        
        // Define center path region (same as wall blocking check)
        val centerPathLeft = 0.3f
        val centerPathRight = 0.7f
        val centerPathTop = 0.3f
        val centerPathBottom = 0.8f
        
        val xStart = (centerPathLeft * (depth[0].size - 1)).toInt()
        val xEnd = (centerPathRight * (depth[0].size - 1)).toInt()
        val yStart = (centerPathTop * (depth.size - 1)).toInt()
        val yEnd = (centerPathBottom * (depth.size - 1)).toInt()
        
        // Sample center path for obstacles
        var minDistance = Float.MAX_VALUE
        var sampleCount = 0
        
        val stepX = max(1, (xEnd - xStart) / 8)
        val stepY = max(1, (yEnd - yStart) / 8)
        
        for (y in yStart..yEnd step stepY) {
            for (x in xStart..xEnd step stepX) {
                if (y in depth.indices && x in depth[0].indices) {
                    val rawDepth = depth[y][x]
                    if (rawDepth > 0f) {
                        val distance = RawDepth(rawDepth).toMeters()
                        if (distance < minDistance) minDistance = distance
                        sampleCount++
                    }
                }
            }
        }
        
        // Consider center path clear if closest obstacle is at least 1.5m away
        return sampleCount > 0 && minDistance >= 1.5f
    }

    private fun smoothWallDetection(currentWallDetected: Boolean, currentDistance: Float?): Pair<Boolean, Float?> {
        // Add current state to history
        wallStateHistory.add(currentWallDetected)
        if (wallStateHistory.size > WALL_STATE_HISTORY_SIZE) {
            wallStateHistory.removeAt(0)
        }
        
        // Add current distance to history
        if (currentDistance != null) {
            wallDistanceHistory.add(currentDistance)
            if (wallDistanceHistory.size > WALL_DISTANCE_HISTORY_SIZE) {
                wallDistanceHistory.removeAt(0)
            }
        }
        
        // Require majority consensus for wall detection (at least 2 out of 3 frames)
        val wallDetectedCount = wallStateHistory.count { it }
        val smoothedWallDetected = wallDetectedCount >= 2
        
        // Average the distance measurements for stability
        val smoothedDistance = if (wallDistanceHistory.isNotEmpty()) {
            wallDistanceHistory.average().toFloat()
        } else null
        
        return Pair(smoothedWallDetected, smoothedDistance)
    }

    private fun mergeAdjacentWallRegions(candidates: List<Pair<RectF, Float>>, bestRegion: RectF): RectF {
        if (candidates.size <= 1) return bestRegion
        
        // Find candidates that are adjacent to the best region (lightweight check)
        val adjacentThreshold = 0.15f // 15% overlap or adjacency
        val mergeableRegions = mutableListOf<RectF>()
        mergeableRegions.add(bestRegion)
        
        for ((candidate, score) in candidates) {
            if (candidate == bestRegion) continue
            
            // Check if regions are adjacent or overlapping
            val horizontalOverlap = !(candidate.right < bestRegion.left - adjacentThreshold || 
                                    candidate.left > bestRegion.right + adjacentThreshold)
            val verticalOverlap = !(candidate.bottom < bestRegion.top - adjacentThreshold || 
                                  candidate.top > bestRegion.bottom + adjacentThreshold)
            
            if (horizontalOverlap && verticalOverlap && score >= WALL_MIN_SCORE_THRESHOLD * 0.8f) {
                mergeableRegions.add(candidate)
            }
        }
        
        // If we found adjacent regions, create a bounding box around all of them
        if (mergeableRegions.size > 1) {
            var minLeft = Float.MAX_VALUE
            var minTop = Float.MAX_VALUE
            var maxRight = -Float.MAX_VALUE
            var maxBottom = -Float.MAX_VALUE
            
            for (region in mergeableRegions) {
                minLeft = min(minLeft, region.left)
                minTop = min(minTop, region.top)
                maxRight = max(maxRight, region.right)
                maxBottom = max(maxBottom, region.bottom)
            }
            
            return RectF(minLeft, minTop, maxRight, maxBottom)
        }
        
        return bestRegion
    }

}

