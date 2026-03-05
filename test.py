from paddleocr import TableRecognitionPipelineV2

pipeline = TableRecognitionPipelineV2()
# ocr = TableRecognitionPipelineV2(use_doc_orientation_classify=True) # 通过 use_doc_orientation_classify 指定是否使用文档方向分类模型
# ocr = TableRecognitionPipelineV2(use_doc_unwarping=True) # 通过 use_doc_unwarping 指定是否使用文本图像矫正模块
# ocr = TableRecognitionPipelineV2(device="gpu") # 通过 device 指定模型推理时使用 GPU
output = pipeline.predict("./table_recognition_v2.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/")
    res.save_to_xlsx("./output/")
    res.save_to_html("./output/")
    res.save_to_json("./output/")