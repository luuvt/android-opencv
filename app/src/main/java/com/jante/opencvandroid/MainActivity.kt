package com.jante.opencvandroid

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.*
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.jante.opencvandroid.databinding.ActivityMainBinding
import kotlinx.coroutines.sync.Mutex
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.imgproc.Imgproc


class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2,
    View.OnTouchListener {
    private lateinit var binding: ActivityMainBinding
    private var cameraView: CameraBridgeViewBase? = null
    private lateinit var currentFrame: Mat
    private var rgba: Mat? = null
    private var detector: ColorBlobDetector? = null
    private var maxRadius: Float = 50.0F
    private var minRadius: Float = 10.0F

    private var mutex = Mutex()
    private var blobList: MutableList<Point> = mutableListOf()

    private var viewMode: Int = 0
    private val VIEW_MODE_RGBA = 0
    private val VIEW_MODE_GRAY = 1
    private val VIEW_MODE_CANNY = 2
    private val VIEW_MODE_FEATURES = 5

    private var itemPreviewRGBA: MenuItem? = null
    private var itemPreviewGray: MenuItem? = null
    private var itemPreviewCanny: MenuItem? = null
    private var itemPreviewFeatures: MenuItem? = null

    private var isStartThread = false

    private val loader = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
                    cameraView?.enableView()
                    cameraView!!.setOnTouchListener(this@MainActivity)
                }
                else -> super.onManagerConnected(status)
            }
        }
    }


    private fun initView() {
        cameraView = findViewById(R.id.open_surface_view)
        cameraView!!.setCvCameraViewListener(this)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)

        initView()
    }

    private fun checkPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestPermission() {
        ActivityCompat.requestPermissions(
            this, arrayOf(Manifest.permission.CAMERA),
            CAMERA_PERMISSION_REQUEST
        )
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        when (requestCode) {
            CAMERA_PERMISSION_REQUEST -> {

                if ((grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED)
                    && grantResults[1] == PackageManager.PERMISSION_GRANTED) {
                    loader.onManagerConnected(LoaderCallbackInterface.SUCCESS)
                } else {
                    Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show()
                }
                return
            }

            else -> {

            }
        }
    }

    companion object {
        const val OpenCVTag = "OpenCV"
        private const val MIN_CONTOUR_AREA              = 0.4
        private const val CAMERA_PERMISSION_REQUEST     = 1
        private const val RADIUS_OF_CIRCLE              = 50
        private const val MAX_SIZE_BLOB_LIST            = 1024 * 2
    }

    override fun onResume() {
        super.onResume()

        if (OpenCVLoader.initDebug()) {
            Log.d(OpenCVTag, "OpenCV successfully loaded")
            loader.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        } else {
            Log.d(OpenCVTag, "OpenCV load failed")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, loader)
        }
    }

    override fun onPause() {
        super.onPause()
        if (cameraView != null) {
            cameraView!!.disableView()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (cameraView != null) {
            cameraView!!.disableView()
        }
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        currentFrame = Mat()
        rgba = Mat(height, width, CvType.CV_8UC4)
        detector = ColorBlobDetector()
        detector!!.setHsvColor(Scalar(80.0, 100.0, 100.0))
        // detector!!.setHsvColor(Scalar(28.0, 37.0, 93.0))
        detector!!.setMinContourArea(MIN_CONTOUR_AREA)
    }

    override fun onCameraViewStopped() {
        currentFrame!!.release()
        rgba!!.release()
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        currentFrame = inputFrame!!.rgba()
        rgba = inputFrame!!.rgba()

        detector!!.process(currentFrame)
        val contours: List<MatOfPoint> = detector!!.getContours()
        Log.i(OpenCVTag, "Contours count: " + contours.size)

        var each: Iterator<MatOfPoint?> = contours.iterator()
        each = contours.iterator()
        val center = Point()
        val radius = FloatArray(1)
        while (each.hasNext()) {
            val contour: MatOfPoint = each.next()
            val currentContour2f = MatOfPoint2f()
            contour.convertTo(currentContour2f, CvType.CV_32FC2)
            Imgproc.minEnclosingCircle(currentContour2f, center, radius)
            if (radius[0].toInt() > maxRadius)
                radius[0] = maxRadius
            if (radius[0].toInt() < minRadius)
                radius[0] = minRadius
            Imgproc.circle(currentFrame, center, radius[0].toInt(), Scalar(255.0, 0.0, 0.0), 2)
            // add to blobList
            addBlob(center)
        }

        return currentFrame
    }

    class MyRunnable(parameter: Any?) : Runnable {
        var rgba: Mat = parameter as Mat
        private var detector: ColorBlobDetector = ColorBlobDetector()
        var initializer = false
        override fun run() {
            if (!initializer) {
                initializer = true
                detector.setHsvColor(Scalar(80.0, 100.0, 100.0))
                detector.setMinContourArea(MIN_CONTOUR_AREA)
            }
            detector!!.process(rgba)
            val contours: List<MatOfPoint> = detector!!.getContours()
            Log.i(OpenCVTag, "Contours count: " + contours.size)
        }
    }

    private fun callThreadToDetect(mat: Mat) {
        val r: Runnable = MyRunnable(mat)
        Thread(r).start()
    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        Log.i(OpenCVTag, "Called onCreateOptionsMenu")
        super.onCreateOptionsMenu(menu);
        itemPreviewRGBA     = menu!!.add("Preview RGBA")
        itemPreviewGray     = menu!!.add("Preview Gray")
        itemPreviewCanny    = menu!!.add("Preview Canny")
        itemPreviewFeatures = menu!!.add("Preview Features")

        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        Log.i(OpenCVTag, "Call onOptionsItemSelected with item: " + item)

        if (item == itemPreviewRGBA) {
            viewMode = VIEW_MODE_RGBA
        } else if (item == itemPreviewGray) {
            viewMode = VIEW_MODE_GRAY
        } else if (item == itemPreviewCanny) {
            viewMode = VIEW_MODE_CANNY
        } else if (item == itemPreviewFeatures) {
            viewMode = VIEW_MODE_FEATURES
        }

        return true
    }

    override fun onTouch(p0: View?, p1: MotionEvent?): Boolean {
        Log.i(OpenCVTag,"onTouch event");
        when (p1?.action) {
            MotionEvent.ACTION_DOWN -> {
                val cols = rgba!!.cols()
                val rows = rgba!!.rows()
                // TODO check x, y on contours
                Log.i(OpenCVTag, "CameraView Width (" + cameraView!!.width + ", Height " + cameraView!!.height + ")");
                Log.i(OpenCVTag, "Mat Cols (" + cols + ", Rows " + rows + ")");

                var xOffset: Int = (cameraView!!.width - cols) / 2
                var yOffset: Int =(cameraView!!.height - rows) / 2

                val x = p1.x - xOffset
                val y = p1.y - yOffset
                Log.i(OpenCVTag, "Touch image coordinates: (" + x + ", " + y + ")");
                if ((x < 0) || (y < 0) || (x > cols) || (y > rows)) return false;

                var (blobDetected, isExists) = isExistsBlobInBlobList(Point(x.toDouble(), y.toDouble()))
                if (isExists) {
                    Log.i(OpenCVTag, "Exists ------------------->>>>>>>>>>>>: (" + blobDetected.x + ", " + blobDetected.y + ")");
                }
                // clear blob cache
                blobList.clear()
            }
        }
        return false
    }

    fun addBlob(point: Point) {
        mutex.tryLock()

        if (blobList.size >= MAX_SIZE_BLOB_LIST) {
            blobList.clear()
        }
        else {
            var each: Iterator<Point?> = blobList.iterator()
            each = blobList.iterator()
            while (each.hasNext()) {
                val p: Point = each.next()
                if (isInside(p, RADIUS_OF_CIRCLE + 20, point)) return
            }
            blobList.add(point)
        }
        mutex.unlock()
    }

    fun isInside(p1: Point, rad: Int, p2: Point): Boolean {
        return (p2.x - p1.x) * (p2.x - p1.x) +
                (p2.y - p1.y) * (p2.y - p1.y) <= rad * rad
    }

    fun isExistsBlobInBlobList(p: Point): Pair<Point, Boolean> {
        var blob: Point = Point()
        var isExists: Boolean = false
        mutex.tryLock()
        var each: Iterator<Point?> = blobList.iterator()
        each = blobList.iterator()
        while (each.hasNext()) {
            blob = each.next()
            if (isInside(blob, RADIUS_OF_CIRCLE, p)) {
                isExists = true
                break
            }
        }
        mutex.unlock()
        return Pair(blob, isExists)
    }
}