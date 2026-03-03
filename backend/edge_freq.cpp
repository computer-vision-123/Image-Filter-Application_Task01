#include "image_utils.hpp"

// Canny Edge Detection

py::bytes apply_canny(const py::bytes &data,
                      double t_low,
                      double t_high,
                      int    kernel_size)
{
    cv::Mat img  = decode_image(data);
    cv::Mat gray = to_grayscale(img);
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(kernel_size, kernel_size), 0);
    cv::Mat edges;
    cv::Canny(blurred, edges, t_low, t_high);
    cv::Mat result;
    cv::cvtColor(edges, result, cv::COLOR_GRAY2BGR);

    return encode_image(result);
}

// Sobel,Prewitt,Roberts Edge Detection

static cv::Mat apply_gradient_kernels(const cv::Mat &gray,
                                      const cv::Mat &Kx,
                                      const cv::Mat &Ky,
                                      int direction,
                                      cv::Point anchor = cv::Point(-1, -1))
{
    cv::Mat gray_f;
    gray.convertTo(gray_f, CV_32F);
    
    cv::Mat Gx, Gy, result_f;

    if (direction == 0) {
        // ONLY calculate vertical edges
        cv::filter2D(gray_f, Gx, CV_32F, Kx, anchor);
        result_f = cv::abs(Gx);           
    } 
    else if (direction == 1) {
        // ONLY calculate horizontal edges
        cv::filter2D(gray_f, Gy, CV_32F, Ky, anchor);
        result_f = cv::abs(Gy);           
    } 
    else {
        // Calculate both ONLY when magnitude is requested
        cv::filter2D(gray_f, Gx, CV_32F, Kx, anchor);
        cv::filter2D(gray_f, Gy, CV_32F, Ky, anchor);
        cv::magnitude(Gx, Gy, result_f);  
    }

    cv::Mat result_8u;
    result_f.convertTo(result_8u, CV_8U);
    
    return result_8u;
}

py::bytes apply_sobel(const py::bytes &data, int direction)
{
    cv::Mat img  = decode_image(data);
    cv::Mat gray = to_grayscale(img);

    cv::Mat Kx = (cv::Mat_<float>(3, 3) <<
        -1,  0,  1,
        -2,  0,  2,
        -1,  0,  1);

    cv::Mat Ky = (cv::Mat_<float>(3, 3) <<
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1);

    cv::Mat result_8u = apply_gradient_kernels(gray, Kx, Ky, direction);

    cv::Mat result_bgr;
    cv::cvtColor(result_8u, result_bgr, cv::COLOR_GRAY2BGR);

    return encode_image(result_bgr);
}

py::bytes apply_prewitt(const py::bytes &data, int direction)
{
    cv::Mat img  = decode_image(data);
    cv::Mat gray = to_grayscale(img);

    cv::Mat Kx = (cv::Mat_<float>(3, 3) <<
        -1,  0,  1,
        -1,  0,  1,
        -1,  0,  1);

    cv::Mat Ky = (cv::Mat_<float>(3, 3) <<
        -1, -1, -1,
         0,  0,  0,
         1,  1,  1);

    cv::Mat result_8u = apply_gradient_kernels(gray, Kx, Ky, direction);
    cv::Mat result_bgr;
    cv::cvtColor(result_8u, result_bgr, cv::COLOR_GRAY2BGR);

    return encode_image(result_bgr);
}

py::bytes apply_roberts(const py::bytes &data, int direction)
{
    cv::Mat img  = decode_image(data);
    cv::Mat gray = to_grayscale(img);

    // 2x2 kernels — diagonal gradient operators
    cv::Mat Kx = (cv::Mat_<float>(2, 2) <<
         1,  0,
         0, -1);

    cv::Mat Ky = (cv::Mat_<float>(2, 2) <<
         0,  1,
        -1,  0);

    cv::Mat result_8u = apply_gradient_kernels(gray, Kx, Ky, direction, cv::Point(0, 0));
    cv::Mat result_bgr;
    cv::cvtColor(result_8u, result_bgr, cv::COLOR_GRAY2BGR);

    return encode_image(result_bgr);
}