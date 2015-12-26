package org.lukey.hadoop.bayes.trainning;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Counters;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * 
 * 输出结果为类名（真实类别）+文件名+类名（分类器预测类别） 计算真实类别的P，R和F1 计算某一个真实类别AAA，如果第一列真实类别为AAA
 * 若第三列类别与真实相同则tp加1， 若第三列类别与真实类别不同则fn加1， 如果第一列不是AAA，并且第三列是AAA则fp加1，否则tn加1.
 * 
 */
public class Predict {
	static enum counter {// 分别表示待测文本的单词数和类别数//未能好好使用
		words_in_file, class_counte
	}
	static Map<String, Double> priorMap = new HashMap<>();// 保存先验概率 ALB 0.375
	static Map<String, Map<String, Double>> conditionMap = new HashMap<>();//条件概率
	static Map<String, Double> notFoundMap = new HashMap<>();//类别中没有单词的概率
	// 保存计算结果，遍历要预测的文件，循环计算不同类别的概率，初识值设为先验概率
	// 每次取value累加（需要用Math.log处理）
	static Map<String, Double> predictMap = new HashMap<String, Double>();//预测的类别和概率
	public static int run(Configuration conf) throws Exception {
		conf.set("mapred.job.tracker", "192.168.190.128:9001");

		Job job = new Job(conf, "predict");
		job.setJarByClass(Predict.class);

		job.setInputFormatClass(PredictInputFormat.class);

		job.setMapperClass(PredictMapper.class);
		job.setReducerClass(PredictReducer.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		String testInput = conf.get("testInput");

		List<Path> paths = MyUtils.getSecondDir(conf, testInput);
		for (Path path : paths) {
			PredictInputFormat.addInputPath(job, path);
		}

		String testOutput = conf.get("testOutput");
		FileOutputFormat.setOutputPath(job, new Path(testOutput));

		int exitCode = job.waitForCompletion(true) ? 0 : 1;
		// 调用计数器，只是用来测试查看
		Counters counters = job.getCounters();
		Counter c1 = counters.findCounter(counter.class_counte);
		System.out.println(c1.getDisplayName() + " : " + c1.getValue());// 测试输出文件所有类别和测集个数试的乘积
		return exitCode;
	}

	public static class PredictMapper extends Mapper<Text, Text, Text, Text> {

		protected void setup(Mapper<Text, Text, Text, Text>.Context context)
				throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			
			String priorPath = conf.get("priorProbality");// 读取先验概率
			priorMap = MyUtils.getProbability(conf, priorPath);//调用方法

			String notFoundPath = conf.get("notFoundPath");// 读取每个类别没有找到单词的概率
			notFoundMap = MyUtils.getProbability(conf, notFoundPath);//调用方法

			String conditionPath = conf.get("conditionPath");// 读取条件概率
			Path condPath = new Path(conditionPath);
			FileSystem fs = condPath.getFileSystem(conf);
			FileStatus[] stats = fs.listStatus(condPath);
			for (FileStatus stat : stats) {
				if (!stat.isDir()) {
					Path filePath = stat.getPath();
					String fileName = filePath.getName();
					String[] temp = fileName.split("-");
					if (temp.length == 3) {
						String className = temp[0];// 得到类别名			
						Map<String, Double> oneMap = new HashMap<>();// 根据文件路径读取文件里面内容保存到map
						oneMap = MyUtils.getProbability(conf, filePath.toString());
						conditionMap.put(className, oneMap);
					}
				}
			}
		}

		protected void map(Text key, Text value, Mapper<Text, Text, Text, Text>.Context context)
				throws IOException, InterruptedException {
			// 遍历preditMap如果对应的类别含有该单词value就称其概率，否在乘以notFoundMap里面的概率
			/* Map<String, Map<String, Double>> conditionMap */
			for (String className : priorMap.keySet()) {// 遍历所有可能的类别
				context.getCounter(counter.class_counte).increment(1);// 每个map都会统计所有的类别
				// 用静态方法计算该文档属于哪个类别的概率
				double p = conditionalProbabilityForClass(value.toString(), className);
				Text prob = new Text(className + "\t" + p);
				context.write(key, prob);//key是类别+文件名
			}
		}

		public static double conditionalProbabilityForClass(String content, String className) {
			 // 计算某个文本属于某一类别的概率// className下面每个单词的概率
			Map<String, Double> condMap = conditionMap.get(className);// class中所有出现单词的概率
			double notFindProbability = notFoundMap.get(className);// class中没有出现单词的概率
			double priorProbability = priorMap.get(className);// class的先验概率
			double pClass = Math.log(priorProbability);// 使初始概率等于先验概率，log处理
			// 遍历测试文本中的所有单词，对每个单词的概率计算log后相加
			StringTokenizer itr = new StringTokenizer(content.toString());
			while (itr.hasMoreTokens()) {
				String word = itr.nextToken();
				if (condMap.containsKey(word)) {// 该类别中含有该单词的条件概率
					pClass += Math.log(condMap.get(word));
				} else {// 该类别不含该单词的概率
					pClass += Math.log(notFindProbability);
				}
			}
			return pClass;// 返回属于类别的条件概率
		}

	}

	public static class PredictReducer extends Reducer<Text, Text, Text, Text> {

		@Override
		protected void reduce(Text key, Iterable<Text> values, Reducer<Text, Text, Text, Text>.Context context)
				throws IOException, InterruptedException {
			// find the max probability
			double maxP = -100000000.0;
			String maxClass = "";
			for (Text value : values) {// 遍历文本的所有属于所有类别的概率
				String[] temp = value.toString().split("\t");
				double p = Double.parseDouble(temp[1]);
				if (p > maxP) {//概率大于最大值
					maxP = p;//将其设成最大值
					maxClass = temp[0];//保存最大值的类别
				}
			}
			context.write(key, new Text(maxClass));
		}
	}
}