/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The sample app shows how to use Vision person segmentation and detect face
 to perform realtime image masking effects.
*/

import UIKit
import Vision
import MetalKit
import AVFoundation
import CoreImage.CIFilterBuiltins

final class ViewController: UIViewController {
    
    // The Vision requests and the handler to perform them.
    private let requestHandler = VNSequenceRequestHandler()
    private var facePoseRequest: VNDetectFaceRectanglesRequest!
    private var segmentationRequest = VNGeneratePersonSegmentationRequest()
    
    // A structure that contains RGB color intensity values.
    private var colors: AngleColors?
    
    @IBOutlet weak var cameraView: MTKView! {
        didSet {
            guard metalDevice == nil else { return }
            setupMetal()
            setupCoreImage()
            setupCaptureSession()
        }
    }
    
    
    // The Metal pipeline.
    public var metalDevice: MTLDevice!
    public var metalCommandQueue: MTLCommandQueue!
    
    // The Core Image pipeline.
    public var ciContext: CIContext!
    public var currentCIImage: CIImage? {
        didSet {
            cameraView.draw()
        }
    }
    
    // The capture session that provides video frames.
    public var session: AVCaptureSession?
    
    // MARK: - ViewController LifeCycle Methods
    
    override func viewDidLoad() {
        super.viewDidLoad()
        intializeRequests()
    }
    
    deinit {
        session?.stopRunning()
    }
    
    // MARK: - Prepare Requests
    
    private func intializeRequests() {
        
        // Create a request to detect face rectangles.
        facePoseRequest = VNDetectFaceRectanglesRequest { [weak self] request, _ in
            guard let face = request.results?.first as? VNFaceObservation else { return }
            // Generate RGB color intensity values for the face rectangle angles.
            self?.colors = AngleColors(roll: face.roll, pitch: face.pitch, yaw: face.yaw)
        }
        facePoseRequest.revision = VNDetectFaceRectanglesRequestRevision3
        
        // Create a request to segment a person from an image.
        segmentationRequest = VNGeneratePersonSegmentationRequest()
        segmentationRequest.qualityLevel = .accurate
        segmentationRequest.outputPixelFormat = kCVPixelFormatType_OneComponent8
    }
    
    // MARK: - Perform Requests
    
    private func processVideoFrame(_ framePixelBuffer: CVPixelBuffer) {
        // Perform the requests on the pixel buffer that contains the video frame.
        try? requestHandler.perform([facePoseRequest, segmentationRequest],
                                    on: framePixelBuffer,
                                    orientation: .right)
        
        // Get the pixel buffer that contains the mask image.
        guard let maskPixelBuffer =
                segmentationRequest.results?.first?.pixelBuffer else { return }
        // Process the images.
        blend(original: framePixelBuffer, mask: maskPixelBuffer)
    }
    
    // MARK: - Process Results
    
    // Performs the blend operation.
    private func blend(original framePixelBuffer: CVPixelBuffer,
                       mask maskPixelBuffer: CVPixelBuffer) {
        
        // Remove the optionality from generated color intensities or exit early.
        guard let colors = colors else { return }
        
        // Create CIImage objects for the video frame and the segmentation mask.
        let originalImage = CIImage(cvPixelBuffer: framePixelBuffer).oriented(.right)
        var maskImage = CIImage(cvPixelBuffer: maskPixelBuffer)
        // Scale the mask image to fit the bounds of the video frame.
        let scaleX = originalImage.extent.width / maskImage.extent.width
        let scaleY = originalImage.extent.height / maskImage.extent.height
        maskImage = maskImage.transformed(by: .init(scaleX: scaleX, y: scaleY))

        // Convert CIImage to CGImage
        let maskCGImage = ciContext.createCGImage(maskImage, from: maskImage.extent)!
        // Convert CGImage to 1D-Array
        let (mask1DArray, imageInfo) = convertImageToArray(fromCGImage: maskCGImage)
        // Convert 1D-Array to CGImage
        let filteredCGImage = convertArrayToImage(fromPixelValues: mask1DArray, fromImageInfo : imageInfo)
        // Convert CGImage to CIImage
        let filteredCIImage = CIImage(cgImage: filteredCGImage!)
        
        // Define RGB vectors for CIColorMatrix filter.
        let vectors = [
            "inputRVector": CIVector(x: 0, y: 0, z: 0, w: colors.red),
            "inputGVector": CIVector(x: 0, y: 0, z: 0, w: colors.green),
            "inputBVector": CIVector(x: 0, y: 0, z: 0, w: colors.blue)
        ]
        
        // Create a colored background image.
        let backgroundImage = maskImage.applyingFilter("CIColorMatrix",
                                                       parameters: vectors)
        
        // Blend the original, background, and mask images.
        let blendFilter = CIFilter.blendWithRedMask()
////        blendFilter.inputImage = originalImage
        blendFilter.backgroundImage = backgroundImage
        blendFilter.maskImage = filteredCIImage
        
        // Set the new, blended image as current.
        currentCIImage = blendFilter.outputImage?.oriented(.left)
    }
}
/**
 *  Description : Convert CGImage to 1D-Array
 *
 *  @param : CGImage to convert
 *  @return :
 *      - $0 : UInt8(0~255) type 1d array(vector)
 *      - $1 : Dictionary<String : Any> : Image information related rendering(width, height, bitsPerComponent, bytesPerRow, totalBytes)
 *
 */
func convertImageToArray(fromCGImage imageRef: CGImage?) -> (pixelValues: [UInt8]?, imageInfo : [String : Any])
{
    var imageInfo : [String : Any] = [:]
    
    var pixelValues: [UInt8]?
    if let imageRef = imageRef {
        let width = imageRef.width
        imageInfo["width"] = width
        
        let height = imageRef.height
        imageInfo["height"] = height
        
        let bitsPerComponent = imageRef.bitsPerComponent
        imageInfo["bitsPerComponent"] = bitsPerComponent
        
        let bytesPerRow = imageRef.bytesPerRow
        imageInfo["bytesPerRow"] = bytesPerRow
        
        let totalBytes = height * bytesPerRow
        imageInfo["totalBytes"] = totalBytes

        let colorSpace = CGColorSpaceCreateDeviceGray()
        var intensities = [UInt8](repeating: 0, count: totalBytes)
        let contextRef = CGContext(data: &intensities, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: 0)
        contextRef?.draw(imageRef, in: CGRect(x: 0.0, y: 0.0, width: CGFloat(width), height: CGFloat(height)))

        pixelValues = intensities
        
        
        // Choose one channel of 4 channel(R,G,B,W)
        
        /**
            @Description : Choose the first channel (R channel of R,G,B,W)
         */
//        for (index, element) in pixelValues!.enumerated() {
//            if(index % 4 == 0){
//                for i in 1...3 {
//                    pixelValues?[index+i] = element
//                }
//            }
//        }
        
        /**
            @Description : Choose the second channel(g channgel of r,g,b,w)
         */
//        for (index, element) in pixelValues!.enumerated() {
//            if(index % 4 == 1){
//                for i in 0...2 {
//                    if(i == 0) {
//                        pixelValues?[index-1] = element
//                    }
//                    else {
//                        pixelValues?[index+i] = element
//                    }
//                }
//            }
//        }
        
        /**
         @Description : Choose the third channel(b channel of r,g,b,w)
        */
//        for (index, element) in pixelValues!.enumerated() {
//            if(index % 4 == 2){
//                for i in 1...3 {
//                    if(i != 3) {
//                        pixelValues?[index-i] = element
//                    }
//                    else {
//                        pixelValues?[index+1] = element
//                    }
//                }
//            }
//        }
        /**
         @Description : Choose the last channel(w channel of r,g,b,w)
        */
//        for (index, element) in pixelValues!.enumerated() {
//            if(index % 4 == 3){
//                for i in 1...3 {
//                    pixelValues?[index-i] = element
//                }
//            }
//        }
        
    }
    
    return (pixelValues, imageInfo)
}

/**
 *  Description : Convert 1D-Array to CGImage
 *
 *  @param :
 *      - $0 : UInt8(0~255) type 1d array(vector)
 *      - $1 : Dictionary<String : Any> : Image information related rendering(width, height, bitsPerComponent, bytesPerRow, totalBytes)
 *  @return : converted CGImage
 *
 */
func convertArrayToImage(fromPixelValues pixelValues: [UInt8]?, fromImageInfo imageInfo : [String : Any]) -> CGImage?
{
    var imageRef: CGImage?
    if var pixelValues = pixelValues {
        imageRef = withUnsafePointer(to: &pixelValues, {
            ptr -> CGImage? in
            var imageRef: CGImage?
            let colorSpaceRef = CGColorSpaceCreateDeviceGray()
            let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue).union(CGBitmapInfo())
            let data = UnsafeRawPointer(ptr.pointee).assumingMemoryBound(to: UInt8.self)
            let releaseData: CGDataProviderReleaseDataCallback = {
                (info: UnsafeMutableRawPointer?, data: UnsafeRawPointer, size: Int) -> () in
            }
            
            if let providerRef = CGDataProvider(dataInfo: nil, data: data, size: imageInfo["totalBytes"] as! Int, releaseData: releaseData) {
                imageRef = CGImage(width: imageInfo["width"] as! Int,
                                   height: imageInfo["height"] as! Int,
                                   bitsPerComponent: imageInfo["bitsPerComponent"] as! Int,
                                   bitsPerPixel: imageInfo["bitsPerComponent"] as! Int,
                                   bytesPerRow: imageInfo["bytesPerRow"] as! Int,
                                   space: colorSpaceRef,
                                   bitmapInfo: bitmapInfo,
                                   provider: providerRef,
                                   decode: nil,
                                   shouldInterpolate: false,
                                   intent: CGColorRenderingIntent.defaultIntent)
            }
            return imageRef
        })
    }

    return imageRef
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Grab the pixelbuffer frame from the camera output
        guard let pixelBuffer = sampleBuffer.imageBuffer else { return }
        processVideoFrame(pixelBuffer)
    }
}
