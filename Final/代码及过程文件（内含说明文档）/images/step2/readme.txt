dr10.gif: drop_rate初始为10递增，根据RGB通道分别抛弃像素，200轮递增的结果。
dr90.gif: drop_rate初始为90递减，根据RGB通道分别抛弃像素，200轮递减的结果。
dr90_results.gif: drop_rate初始为90递减，综合3通道抛弃率，直接抛弃整个像素（0-1个通道抛弃则忽略，2-3个通道抛弃则抛弃），结合两重早停的demo。
dr90_test.gif: drop_rate初始为90递减，综合3通道抛弃率，直接抛弃整个像素（0-1个通道抛弃则忽略，2-3个通道抛弃则抛弃），200轮递减的结果。
optimize_display_dr10,jpg: dr10的resnet50识别概率分布情况。
optimize_display_dr90.jpg: dr10的resnet90识别概率分布情况。
optimize_display_dr90_test.jpg: dr10的resnet90_test识别概率分布情况。
origin_image.jpg: 上面实验小猫图的原图。
result_3.jpg; result_7.jpg: dr90_results实验中无早停条件的两个比较离谱的结果。
文件夹里面都是对应gif的源素材。