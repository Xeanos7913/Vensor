#include <vector>
#include <string>
#include <filesystem>
#include <stdexcept>
#include "Tensor.hpp"
#include "VkMemAlloc.hpp"

template<typename T>
struct ImageDataloader {
    TensorPool<T>* tensorPool;
    std::vector<Tensor<T>*> imagesBatchTensors; // num_batches * {B, C, H, W}
    std::vector<Tensor<T>*> labelTensors; // num_batches * {B, 1, num_class} one hot vectors
    uint32_t batch_size;
    uint32_t num_batches;
    size_t num_pixels;

    size_t height, width, channels;

    ImageDataloader() {};

    ImageDataloader(TensorPool<T>* pool, uint32_t batch_size, uint32_t img_width, uint32_t img_height, uint32_t channels, uint32_t num_batches)
        : tensorPool(pool), batch_size(batch_size), num_batches(num_batches), width(img_width), height(img_height), channels(channels)
    {
        // tensor shape now NCHW
        std::vector<uint32_t> shape = { batch_size, channels, img_height, img_width };
        for (uint32_t i = 0; i < num_batches; ++i) {
            Tensor<T>* tensor = &tensorPool->createTensor(shape, "image_batch_tensor_" + std::to_string(i));
            imagesBatchTensors.push_back(tensor);
        }
        num_pixels = channels * img_width * img_height;
    }

    void loadImagesIntoTensor(const std::vector<std::string>& paths, uint32_t batch_index) {
        if (batch_index >= imagesBatchTensors.size())
            throw std::runtime_error("Invalid batch index");

        // map to CPU memory as type T
        T* map = reinterpret_cast<T*>(imagesBatchTensors[batch_index]->dataBuffer->memMap);
        size_t single_image_pixels = height * width;
        size_t single_image_size = channels * single_image_pixels; // C*H*W

        for (uint32_t k = 0; k < paths.size(); ++k) {
            int img_w, img_h, img_c;
            unsigned char* data = stbi_load(paths[k].c_str(), &img_w, &img_h, &img_c, static_cast<int>(channels));
            if (!data)
                throw std::runtime_error("Failed to load image: " + paths[k]);

            if (img_w != static_cast<int>(width) || img_h != static_cast<int>(height))
            {
                stbi_image_free(data);
                throw std::runtime_error("Image size mismatch for: " + paths[k]);
            }

            // destination start for k-th image in the batch (NCHW)
            T* dst_base = map + static_cast<size_t>(k) * single_image_size;

            // stbi returns data in HWC order, convert to CHW
            for (uint32_t c = 0; c < channels; ++c) {
                size_t channel_offset = static_cast<size_t>(c) * single_image_pixels;
                for (size_t y = 0; y < height; ++y) {
                    for (size_t x = 0; x < width; ++x) {
                        size_t src_idx = (y * width + x) * channels + c; // HWC
                        size_t dst_idx = channel_offset + y * width + x; // CHW within one image
                        dst_base[dst_idx] = static_cast<T>(data[src_idx]) / static_cast<T>(255);
                    }
                }
            }

            stbi_image_free(data);
        }

        imagesBatchTensors[batch_index]->dataBuffer->to_gpu();
    }

    // load a num_batches * batch_size number of images into gpu-resident tensors used as input for training one epoch
    void loadImagesFromDirectory(const std::string& directory_path) {
        // Collect all image paths from the directory
        std::vector<std::string> all_image_paths;
        for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
            if (entry.is_regular_file()) {
                all_image_paths.push_back(entry.path().string());
            }
        }

        // Check if there are enough images to populate the tensors
        size_t total_images_needed = imagesBatchTensors.size() * batch_size;
        if (all_image_paths.size() < total_images_needed) {
            throw std::runtime_error("Not enough images in the directory to populate the tensors. "
                                     "Required: " + std::to_string(total_images_needed) + ", Found: " + std::to_string(all_image_paths.size()));
        }

        // Load images into tensors batch by batch
        for (size_t batch_idx = 0; batch_idx < imagesBatchTensors.size(); ++batch_idx) {
            std::vector<std::string> batch_image_paths;
            size_t start_idx = batch_idx * batch_size;
            size_t end_idx = start_idx + batch_size;

            for (size_t i = start_idx; i < end_idx; ++i) {
                batch_image_paths.push_back(all_image_paths[i]);
            }

            loadImagesIntoTensor(batch_image_paths, static_cast<uint32_t>(batch_idx));
        }
    }
};

template<typename T>
struct MNISTDataloader {
    TensorPool<T>* tensorPool;
    std::vector<Tensor<T>*> imagesBatchTensors; // num_batches * {B, C, H, W}
    std::vector<Tensor<T>*> labelTensors;       // num_batches * {B, 1, num_classes}
    uint32_t batch_size;
    uint32_t num_batches;
    uint32_t num_classes = 10;
    uint32_t height = 28;
    uint32_t width = 28;
    uint32_t channels = 1;
    size_t num_pixels;

    MNISTDataloader() = default;

    MNISTDataloader(TensorPool<T>* pool, uint32_t batch_size, uint32_t num_batches, const std::string& name = "mnist_image_bath_")
        : tensorPool(pool), batch_size(batch_size), num_batches(num_batches)
    {
        num_pixels = channels * height * width;
        // NCHW shapes
        std::vector<uint32_t> img_shape = { batch_size, channels, height, width };
        std::vector<uint32_t> lbl_shape = { batch_size, 1, num_classes };

        for (uint32_t i = 0; i < num_batches; ++i) {
            auto& img = tensorPool->createTensor(img_shape, name + "inputs" + std::to_string(i));
            auto& lbl = tensorPool->createTensor(lbl_shape, name + "labels" + std::to_string(i));
            imagesBatchTensors.push_back(&img);
            labelTensors.push_back(&lbl);
        }
    }

private:
    static uint32_t readBE32(std::ifstream& f) {
        uint8_t b[4];
        f.read(reinterpret_cast<char*>(b), 4);
        return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
    }

public:
    void loadMNIST(const std::string& image_path, const std::string& label_path) {
        std::ifstream imgFile(image_path, std::ios::binary);
        std::ifstream lblFile(label_path, std::ios::binary);
        if (!imgFile || !lblFile)
            throw std::runtime_error("Failed to open MNIST ubyte files");

        uint32_t imgMagic = readBE32(imgFile);
        uint32_t numImgs  = readBE32(imgFile);
        uint32_t rows     = readBE32(imgFile);
        uint32_t cols     = readBE32(imgFile);

        uint32_t lblMagic = readBE32(lblFile);
        uint32_t numLbls  = readBE32(lblFile);

        if (imgMagic != 2051 || lblMagic != 2049)
            throw std::runtime_error("Invalid MNIST magic numbers");
        if (numImgs != numLbls)
            throw std::runtime_error("Image/label count mismatch");

        size_t total_images_needed = static_cast<size_t>(batch_size) * num_batches;
        if (numImgs < total_images_needed)
            throw std::runtime_error("Not enough MNIST samples for configured dataloader");

        std::vector<uint8_t> imageBuf(rows * cols);
        for (uint32_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            T* imageMap = reinterpret_cast<T*>(imagesBatchTensors[batch_idx]->dataBuffer->memMap);
            T* labelMap = reinterpret_cast<T*>(labelTensors[batch_idx]->dataBuffer->memMap);

            for (uint32_t i = 0; i < batch_size; ++i) {
                size_t global_idx = static_cast<size_t>(batch_idx) * batch_size + i;

                // read single image
                imgFile.read(reinterpret_cast<char*>(imageBuf.data()), rows * cols);
                uint8_t lbl;
                lblFile.read(reinterpret_cast<char*>(&lbl), 1);

                // write into NCHW layout; MNIST is single-channel
                T* dst_base = imageMap + static_cast<size_t>(i) * num_pixels;
                // channel 0 contiguous, normalize to [-1, 1]
                for (size_t j = 0; j < rows * cols; ++j)
                    dst_base[j] = (static_cast<T>(imageBuf[j]) / static_cast<T>(127.5)) - 1.0f;

                // one-hot label
                std::fill(labelMap + i * num_classes, labelMap + (i + 1) * num_classes, static_cast<T>(0));
                labelMap[i * num_classes + lbl] = static_cast<T>(1);
            }

            imagesBatchTensors[batch_idx]->dataBuffer->to_gpu();
            labelTensors[batch_idx]->dataBuffer->to_gpu();
        }
    }
};