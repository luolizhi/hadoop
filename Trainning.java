package org.lukey.hadoop.bayes.trainning;

import org.apache.hadoop.conf.Configuration;


/**
 * 
 * 测试类，负责传递所有的输入输出路径，以及中转路径，通过configuration传递
 * main函数
 * 
 */
public class Trainning {

	public static void main(String[] args) throws Exception {
		

		Configuration conf = new Configuration();
		conf.set("mapred.jar", "E://eclipse//jar-work//Trainning.jar");	//在eclipse下用ant打包使用
		conf.set("mapred.job.tracker", "192.168.190.128:9001");
		
		//设置一个常数FILENUMBER，计算文本数不低于该常数的类别
		conf.setInt("FILENUMBER", 50);//注释该行后，取默认的数字10
		
		// 设置单词先验概率的保存路径
		String priorProbality = "hdfs://192.168.190.128:9000/user/hadoop/output/priorP/priorProbability.txt";
		conf.set("priorProbality", priorProbality);

		// 单词总种类数的保存路径，优化后这个不用写文件，直接保存在conf里面
		String totalWordsPath = "hdfs://192.168.190.128:9000/user/hadoop/output/totalwords.txt";
		conf.set("totalWordsPath", totalWordsPath);

		// 每个类别中单词总数
		String wordsInClassPath = "hdfs://192.168.190.128:9000/user/hadoop/mid/wordsFrequence/_wordsInClass/wordsInClass-r-00000";
		conf.set("wordsInClassPath", wordsInClassPath);

		// 设置输入 和 单词词频的输出路径
		String input = "hdfs://192.168.190.128:9000/user/hadoop/input/NBCorpus/Country";
		//调试程序所用的少量文本集，调试时注释掉23行
//		String input = "hdfs://192.168.190.128:9000/user/hadoop/test/";	
		
		String wordsOutput = "hdfs://192.168.190.128:9000/user/hadoop/mid/wordsFrequence";
		conf.set("input", input);
		conf.set("wordsOutput", wordsOutput);

		// 每个类别单词概率保存路径,
		// 单词词频的输入路径也就是单词词频的输出路径
		String conditionPath = "hdfs://192.168.190.128:9000/user/hadoop/output/probability/";
		conf.set("conditionPath", conditionPath);
	
		//predict
		//测试的输入输出目录
		String testInput = "hdfs://192.168.190.128:9000/user/hadoop/test";
		conf.set("testInput", testInput);
		
		String testOutput = "hdfs://192.168.190.128:9000/user/hadoop/output/Predict";
		conf.set("testOutput", testOutput);
		
		//条件概率conditionPath和先验概率priorProbality
		//上面已经设置好了

		String notFoundPath = "hdfs://192.168.190.128:9000/user/hadoop/output/probability/_notFound/notFound-m-00000";
		conf.set("notFoundPath", notFoundPath);
		

		//评估分类器的好坏
		String result = "hdfs://192.168.190.128:9000/user/hadoop/output/Predict/part-r-00000";
		conf.set("result", result);
		String resultOut = "hdfs://192.168.190.128:9000/user/hadoop/output/Result";
		conf.set("resultOut", resultOut);
		
		System.out.println("-------Start-------");
		FileCount.run(conf);//统计文件个数，计算先验概率
		System.out.println("------FileCount OK------");
		
		WordCount.run(conf);//统计单词个数
		Probability.run(conf);//计算条件概率，即每个单词在各个类别中出现的概率
		System.out.println("------Probability OK------");
		
		Predict.run(conf);//预测测试文本的类别
		System.out.println("------Predict OK------");
		
		Evaluation.run(conf);//评估预测的效果
		
		System.out.println("------Evaluation OK------");
		System.out.println("------All Over------");
	
	}

}
