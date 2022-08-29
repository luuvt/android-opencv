package com.jante.opencvandroid

import org.opencv.android.*
import org.opencv.core.*
import org.opencv.imgproc.Imgproc


class ColorBlobDetector {
    // Lower and Upper bounds for range checking in HSV color space
    private val mLowerBound: Scalar = Scalar(0.0, 0.0, 0.0)
    private val mUpperBound: Scalar = Scalar(0.0, 0.0, 0.0)

    // Color radius for range checking in HSV color space
    private var mColorRadius = Scalar(25.0, 50.0, 50.0, 0.0)
    private val mSpectrum = Mat()
    private val mContours: MutableList<MatOfPoint> = ArrayList()

    // Cache
    var mPyrDownMat = Mat()
    var mHsvMat = Mat()
    private var mMask = Mat()
    var mDilatedMask = Mat()
    var mHierarchy = Mat()

    fun setColorRadius(radius: Scalar) {
        mColorRadius = radius
    }

    fun setHsvColor(hsvColor: Scalar) {
        val minH: Double =
            if (hsvColor.`val`[0] >= mColorRadius.`val`[0]) hsvColor.`val`[0] - mColorRadius.`val`[0] else 0.0
        val maxH: Double =
            if (hsvColor.`val`[0] + mColorRadius.`val`[0] <= 255) hsvColor.`val`[0] + mColorRadius.`val`[0] else 255.0
        mLowerBound.`val`[0] = minH
        mUpperBound.`val`[0] = maxH
        mLowerBound.`val`[1] = hsvColor.`val`[1] - mColorRadius.`val`[1]
        mUpperBound.`val`[1] = hsvColor.`val`[1] + mColorRadius.`val`[1]
        mLowerBound.`val`[2] = hsvColor.`val`[2] - mColorRadius.`val`[2]
        mUpperBound.`val`[2] = hsvColor.`val`[2] + mColorRadius.`val`[2]
        mLowerBound.`val`[3] = 0.0
        mUpperBound.`val`[3] = 255.0
        val spectrumHsv = Mat(1, (maxH - minH).toInt(), CvType.CV_8UC3)
        var j = 0
        while (j < maxH - minH) {
            val tmp = byteArrayOf((minH + j).toInt().toByte(), 255.toByte(), 255.toByte())
            spectrumHsv.put(0, j, tmp)
            j++
        }
        Imgproc.cvtColor(spectrumHsv, mSpectrum, Imgproc.COLOR_HSV2RGB_FULL, 4)
    }

    fun setMinContourArea(area: Double) {
        mMinContourArea = area
    }

    fun getSpectrum(): Mat {
        return mSpectrum
    }

    fun process(rgbaImage: Mat?) {
        Imgproc.pyrDown(rgbaImage, mPyrDownMat)
        Imgproc.pyrDown(mPyrDownMat, mPyrDownMat)
        Imgproc.cvtColor(mPyrDownMat, mHsvMat, Imgproc.COLOR_RGB2HSV_FULL)
        Core.inRange(mHsvMat, mLowerBound, mUpperBound, mMask)
        var kernel: Mat = Mat.ones(5, 5, CvType.CV_8U)
        Imgproc.dilate(mMask, mDilatedMask, kernel)
        val contours: List<MatOfPoint> = ArrayList()
        Imgproc.findContours(
            mDilatedMask,
            contours,
            mHierarchy,
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        // Find max contour area
        var maxArea: Double = 0.0
        var each: Iterator<MatOfPoint> = contours.iterator()
        while (each.hasNext()) {
            val wrapper = each.next()
            val area = Imgproc.contourArea(wrapper)
            if (area > maxArea) maxArea = area
        }

        // Filter contours by area and resize to fit the original image size
        mContours.clear()
        each = contours.iterator()
        while (each.hasNext()) {
            val contour = each.next()
            if (Imgproc.contourArea(contour) > mMinContourArea * maxArea) {
                Core.multiply(contour, Scalar(4.0, 4.0), contour)
                mContours.add(contour)
            }
        }
    }

    public fun getContours(): List<MatOfPoint> {
        return mContours
    }

    companion object {
        // Minimum contour area in percent for contours filtering
        private var mMinContourArea = 0.1
    }
}
