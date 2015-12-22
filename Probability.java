package org.lukey.hadoop.bayes.trainning;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;

public class Probability {

	// 试着用LOG打印调试信息
	private static final Log LOG = LogFactory.getLog(FileInputFormat.class);
	public static int total = 0; // 所有类中单词的总类别数
	private static MultipleOutputs<Text, DoubleWritable> mos;

	public static int run(Configuration conf) throws Exception {

		// 先读取单词总类别数
		total = conf.getInt("TOTALWORDS", 0);
		LOG.info("------>total = " + total);// 读取所有类别的单词总种类数，只为在控制台显示单词总数

		System.out.println("total ==== " + total);// 也可以在控制台显示打印信息

		Job job = new Job(conf, "probability");

		job.setJarByClass(Probability.class);

		job.setMapperClass(ProbMapper.class);
		job.setReducerClass(ProbReducer.class);

		String input = conf.get("wordsOutput");
		String output = conf.get("conditionPath");

		FileInputFormat.addInputPath(job, new Path(input));
		FileOutputFormat.setOutputPath(job, new Path(output));

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(DoubleWritable.class);

		int exitCode = job.waitForCompletion(true) ? 0 : 1;
		return exitCode;

	}

	// Mapper
	static class ProbMapper extends Mapper<LongWritable, Text, Text, DoubleWritable> {

		private static DoubleWritable number = new DoubleWritable();
		private static Text className = new Text();
		
		private static Map<String, Integer> fileMap = new HashMap<String, Integer>();// 保存类别中单词总数
		
		protected void setup(Mapper<LongWritable, Text, Text, DoubleWritable>.Context context)
				throws IOException, InterruptedException {
			mos = new MultipleOutputs<Text, DoubleWritable>(context);//文件的多文件名输出
			
			Configuration conf = context.getConfiguration();
			String filePath = conf.get("wordsInClassPath");//获取类别中单词总数的文件路径
			FileSystem fs = FileSystem.get(URI.create(filePath), conf);//获取文件系统对象
			FSDataInputStream inputStream = fs.open(new Path(filePath));//定义文件输入流对象
			BufferedReader buffer = new BufferedReader(new InputStreamReader(inputStream));
			String strLine = null;
			while ((strLine = buffer.readLine()) != null) {
				String[] temp = strLine.split("\t");
				fileMap.put(temp[0], Integer.parseInt(temp[1]));
			}
		}
	
		protected void map(LongWritable key, Text value,
				Mapper<LongWritable, Text, Text, DoubleWritable>.Context context)
						throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			int tot = conf.getInt("TOTALWORDS", 0);// map中需要重新获取单词总种类数

			// System.out.println("total = " + total);//total = 0无法使用，只能通过conf传值
			// System.out.println("tot = " + tot);//这个输出是在userlog中的stdout中显示

			// 输入的格式如下：
			// ALB weekend 1
			// ALB weeks 3
			Map<String, Map<String, Integer>> baseMap = new HashMap<String, Map<String, Integer>>(); // 保存基础数据，<类名,<单词，词频>>
			// Map<String, Map<String, Double>> priorMap = new HashMap<String,
			// Map<String, Double>>(); // 保存每个单词出现的概率

			String[] temp = value.toString().split("\t");
			// 先将数据存到baseMap中
			if (temp.length == 3) {
				// 文件夹名类别名
				if (baseMap.containsKey(temp[0])) {
					baseMap.get(temp[0]).put(temp[1], Integer.parseInt(temp[2]));
				} else {
					Map<String, Integer> oneMap = new HashMap<String, Integer>();
					oneMap.put(temp[1], Integer.parseInt(temp[2]));
					baseMap.put(temp[0], oneMap);
				}
			} // 读取数据完毕，全部保存在baseMap中

			int allWordsInClass = 0;

			for (Map.Entry<String, Map<String, Integer>> entries : baseMap.entrySet()) { // 遍历类别
				allWordsInClass = fileMap.get(entries.getKey());
				for (Map.Entry<String, Integer> entry : entries.getValue().entrySet()) { // 遍历类别中的单词词频求概率
					double p = (entry.getValue() + 1.0) / (allWordsInClass + tot);// 避免出现0，每个单词都加1

					className.set(entries.getKey() + "\t" + entry.getKey());
					number.set(p);
//					LOG.info("------>p = " + p);//这里的输出是在userlog中的syslog中不会显示出来
					mos.write(new Text(entry.getKey()), number, entries.getKey());// 将每个类别的单词词频分别写入其类名命名的文件中
					// context.write(className, number);
				}
			}
		}
	
		protected void cleanup(Mapper<LongWritable, Text, Text, DoubleWritable>.Context context)
				throws IOException, InterruptedException {
			// 最后计算类别中不存在单词的概率，每个类别都是一个常数
			Configuration conf = context.getConfiguration();
			int tot = conf.getInt("TOTALWORDS",0);//获取所有类别中单词总数
			for (Map.Entry<String, Integer> entry : fileMap.entrySet()) { // 遍历各个类别中单词的总数，得到所有的类别
				double notFind = (1.0) / (entry.getValue() + tot);
				number.set(notFind);
				mos.write(new Text(entry.getKey()), number, "_notFound" + "\\" + "notFound");//文件夹前加_可以让后续的mapreduce忽略该文件夹

			}
			mos.close();
		}
	}

	// Reducer
	static class ProbReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
		//什么也不用做
		
		@Override
		protected void reduce(Text arg0, Iterable<DoubleWritable> arg1,
				Reducer<Text, DoubleWritable, Text, DoubleWritable>.Context arg2)
						throws IOException, InterruptedException {
			super.reduce(arg0, arg1, arg2);
		}

	/*	DoubleWritable result = new DoubleWritable();
	  protected void reduce(Text key, Iterable<DoubleWritable> values,
				Reducer<Text, DoubleWritable, Text, DoubleWritable>.Context context)
						throws IOException, InterruptedException {
			double sum = 0L;
			for (DoubleWritable value : values) {
				sum += value.get();
			}
			result.set(sum);
			context.write(key, result);// 只是做简单的归并
		}*/
	}
}
