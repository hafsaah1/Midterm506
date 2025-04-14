#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <vector>

int main() {
    // Open webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the webcam.\n";
        return -1;
    }

    // QR code detector
    cv::QRCodeDetector qrDetector;
    std::vector<std::string> detectedCommands;
    std::string lastCommand = "";

    std::cout << "Show QR code cards (left, right, forward, backwards). Press 'q' to quit.\n";

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        std::string decoded;
        std::vector<cv::Point> points;
        decoded = qrDetector.detectAndDecode(frame, points);

        if (!decoded.empty() && decoded != lastCommand) {
            std::cout << "Detected: " << decoded << "\n";
            detectedCommands.push_back(decoded);
            lastCommand = decoded;

            // Draw bounding box
            if (points.size() == 4) {
                for (int i = 0; i < 4; ++i) {
                    cv::line(frame, points[i], points[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
                }
            }
        }

        // Display
        cv::imshow("QR Command Reader", frame);

        // Break on 'q' key
        char key = (char)cv::waitKey(1);
        if (key == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();

    std::cout << "\nFinal command sequence:\n[ ";
    for (const auto& cmd : detectedCommands) {
        std::cout << cmd << " ";
    }
    std::cout << "]\n";

    return 0;
}
