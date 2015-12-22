package org.lukey.hadoop.bayes.trainning;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Counters;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


/**
 * 
 * 输出结果为类名（真实类别）+文件名+类名（分类器预测类别）
 * 计算真实类别的P，R和F1
 * 计算某一个真实类别AAA，如果第一列真实类别为AAA
 * 若第三列类别与真实相同则tp加1，
 * 若第三列类别与真实类别不同则fn加1，
 * 如果第一列不是AAA，并且第三列是AAA则fp加1，否则tn加1.
 * 
 */
public class Predict {
	
//	 private static final Log LOG = LogFactory.getLog(Predict.class);

	static enum counter {// 分别表示待测文本的单词数和类别数//未能好好使用
		words_in_file, class_counte
	}
	
	static Map<String, Double> priorMap = new HashMap<>();//保存先验概率 ALB 0.375

	// 每个类别里面单词的概率 abassi 3.476341324572953E-5 类别名是文件名，需要特殊处理
	// 初步想法是将文件名解析出来，作为map的key，里面的数据作为map的value（单词 概率）
	static Map<String, Map<String, Double>> conditionMap = new HashMap<>();

	// 每个类别里面没有找到的单词的概率ALB 4.062444649191655E-7
	static Map<String, Double> notFoundMap = new HashMap<>();

	// 保存计算结果，遍历要预测的文件，循环计算不同类别的概率，初识值可以设为先验概率
	// 先验概率 ALB 0.375（key用文件名+分类名）
	// 每次取value累成（需要用log处理）
	static Map<String, Double> predictMap = new HashMap<String, Double>();

	public static int run(Configuration conf) throws Exception  {
		conf.set("mapred.job.tracker", "192.168.190.128:9001");
		
		Job job = new Job(conf, "predict");
		job.setJarByClass(Predict.class);

		job.setInputFormatClass(WholeFileInputFormat.class);

		job.setMapperClass(PredictMapper.class);
		job.setReducerClass(PredictReducer.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		String testInput = conf.get("testInput");
		
		List<Path> paths = MyUtils.getSecondDir(conf, testInput);
		for (Path path : paths) {
			WholeFileInputFormat.addInputPath(job, path);
		}

		String testOutput = conf.get("testOutput");
		FileOutputFormat.setOutputPath(job, new Path(testOutput));

		int exitCode = job.waitForCompletion(true) ? 0 : 1;

		// 调用计数器
		Counters counters = job.getCounters();
		Counter c1 = counters.findCounter(counter.class_counte);
		System.out.println(c1.getDisplayName() + " : " + c1.getValue());

		return exitCode;
	}

	// map
	public static class PredictMapper extends Mapper<LongWritable, Text, Text, Text> {

		protected void setup(Mapper<LongWritable, Text, Text, Text>.Context context)
				throws IOException, InterruptedException {
			
			Configuration conf = context.getConfiguration();
			
			String priorPath = conf.get("priorProbality");// 读取先验概率
			priorMap = MyUtils.getProbability(conf, priorPath);
			
			String notFoundPath = conf.get("notFoundPath");// 读取每个类别没有找到单词的概率
			notFoundMap = MyUtils.getProbability(conf, notFoundPath);
						
			String conditionPath = conf.get("conditionPath");//读取条件概率
			try {
				conditionMap = MyUtils.getConditionMap(conf, conditionPath);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
			
		protected void map(LongWritable key, Text value, Mapper<LongWritable, Text, Text, Text>.Context context)
				throws IOException, InterruptedException {

			FileSplit split = (FileSplit)context.getInputSplit();//从context获取split，并强转成fileSplit
			Path file = split.getPath();//获取文件输入路径
			String fileName = file.getName(); // 得到文件名，输出的时候写入context
			String trueClassName = file.getParent().getName() ;//得到真实类别
					
			// 遍历preditMap如果对应的类别含有该单词value就称其概率，否在乘以notFoundMap里面的概率
			/* Map<String, Map<String, Double>> conditionMap */

			for (String className : priorMap.keySet()) {//遍历所有可能的类别
				context.getCounter(counter.class_counte).increment(1);//每个map都会统计所有的类别
				// 用静态方法计算该文档属于哪个类别的概率
				double p = conditionalProbabilityForClass(value.toString(), className);
//				LOG.info(className + "------->" + p);//将一个文本所有类别下计算的概率写入log中
//				System.out.println(className + "------->" + p);//输出在log中的stdout中
				Text prob = new Text(className + "\t" + p);
				context.write(new Text(trueClassName + "\t"+ fileName), prob);
			}
		}

		public static double conditionalProbabilityForClass(String content, String className) {
			/*
			 * 计算某个文本属于某一类别的概率
			 */
			// className下面每个单词的概率
			Map<String, Double> condMap = conditionMap.get(className);//class中所有出现单词的概率
			double notFindProbability = notFoundMap.get(className);//class中没有出现单词的概率
			double priorProbability = priorMap.get(className);//class的先验概率
			double pClass = Math.log(priorProbability);//使初始概率等于先验概率，log处理
			//遍历测试文本中的所有单词，对每个单词的概率计算log后相加
			StringTokenizer itr = new StringTokenizer(content.toString());
			while (itr.hasMoreTokens()) {
				String word = itr.nextToken();
				if (condMap.containsKey(word)) {//该类别中含有该单词的条件概率
					pClass += Math.log(condMap.get(word));
				} else {//该类别不含该单词的概率
					pClass += Math.log(notFindProbability);
				}
			}
			return pClass;//返回属于类别的条件概率
		}

	}

	// reduce
	public static class PredictReducer extends Reducer<Text, Text, Text, Text> {

		Double maxP = 0.0;
		@Override
		protected void reduce(Text key, Iterable<Text> values, Reducer<Text, Text, Text, Text>.Context context)
				throws IOException, InterruptedException {
			// find the max probability
			double maxP = -100000000.0;
			String maxClass = "";
			for (Text value : values) {//遍历文本的所有属于所有类别的概率
				String[] temp = value.toString().split("\t");
				double p = Double.parseDouble(temp[1]);
				maxClass = (p>maxP)? temp[0]: maxClass;//double比大小可以直接比，相等要用abs小于一个很小的数
			}
			context.write(key, new Text(maxClass));
		}
	}

}