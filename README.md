# CV_Identify_Insects
9517 detect and identify insects

（咳咳，敲黑板）
这是一个关于检测和识别昆虫的CV项目    
~~虽然只是为了拿分~~    
但是我们还是要好好加油拿高分    
    
项目成员包括两位美丽优雅的女士，Yarra和Rachel        
还有三位帅气认真的男士，Dylon, GW 和 Brett（也就是本人~）    

## 项目方案
（每个部分到时候认领了回来各自完善文案哈） 
### 1. 经典机器学习方法    
*   检测器 (Detector): 滑动窗口 (Sliding Window)。
*   分类器 (Classifier): HOG 特征 + 支持向量机 (SVM)。


### 2. 两阶段深度学习检测器 (Faster R-CNN)
*   检测器 (Detector): 区域提议网络 (RPN)
*   分类器 (Classifier): 网络分类头 (Network Head)

### 3. 单阶段深度学习检测器 (YOLO 或 SSD)
*   检测器 (Detector): 单次网格预测 (Single-Shot Grid)
*   分类器 (Classifier): 集成的分类预测

### 4. 迁移学习
这个就纯靠探索了

## 总结
目前就这么多，后面慢慢维护吧，大家加油
