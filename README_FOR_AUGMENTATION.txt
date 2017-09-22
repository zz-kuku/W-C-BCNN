# README FOR DO_AUGMENTATION

1.给出图片的路径名文件images.txt, 标签分类文件image_class_label.txt, 训练测试验证集
  分割文件train_test_split.txt, 文件内具体格式见参考，可自行修改。
2.只对训练集做扩增，可修改line24:index_Tr，来选择是否对验证和测试集做扩增(选择测试时，
  要注释index_Te)。imageSet(1.train 2.validation 3.test)
3.可自行设定每类样本数的上下限，Maxnum & Minnum。
4.可在line63设定扩增的方式，分为切片、加色、旋转、加躁、遮挡，以及加躁和遮挡的混合。
5.图片保存路径为 [mdir, 'images', imageNames]。

