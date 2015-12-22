package org.lukey.hadoop.bayes.trainning;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Counters;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;

public class WordCount {
	//试着用LOG打印调试信息
	private static final Log LOG = LogFactory.getLog(FileInputFormat.class);

	//将结果输出到不同的文件名中
	private static MultipleOutputs<Text, IntWritable> mos;
	//使用计数器，前面两个没有用好
	static enum WordsNature {
		CLSASS_NUMBER, CLASS_WORDS, TOTALWORDS
	}

	// map
	static class WordCountMapper extends Mapper<Text, Text, Text, IntWritable> {

		private final static IntWritable one = new IntWritable(1);
		private final static IntWritable zero = new IntWritable(0);

		private Text countryName = new Text();
	
		@Override
		protected void map(Text key, Text value, Mapper<Text, Text, Text, IntWritable>.Context context)
				throws IOException, InterruptedException {
//			System.out.println("-00--before map" + key + ":-----");//在logs/userlogs中的stdout输出
			LOG.info("--map000--- " + key + "---------");	//在logs/userlogs中的syslog输出
			StringTokenizer itr = new StringTokenizer(value.toString());
			
			while (itr.hasMoreTokens()) {
				String word = itr.nextToken();
				
				if (!(MyUtils.hasDigit(word) || word.contains("."))) { // 去掉无意义词
					countryName.set(key.toString() + "\t" + word);
					
					context.write(countryName, one); // 统计每个类别中的单词个数 ABL have 1
					context.write(key, one); // 统计类别中的单词总数
					context.write(new Text(word), zero); // 统计单词总数				
				}				
			}
//			System.out.println("-00------我结束了 map---over" + key + ":");
		}
	}

	// Reducer
	static class WordCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
	

		// result 表示每个类别中每个单词的个数
		IntWritable result = new IntWritable();
		Map<String, List<String>> classMap = new HashMap<String, List<String>>();
		Map<String, List<String>> fileMap = new HashMap<String, List<String>>();

		@Override
		protected void reduce(Text key, Iterable<IntWritable> values,
				Reducer<Text, IntWritable, Text, IntWritable>.Context context)
						throws IOException, InterruptedException {
			System.out.println("reduce reduce");
//			LOG.info("---reduce-- " + key);

			int sum = 0;
			for (IntWritable value : values) {
				sum += value.get();
			}

//			result.set(sum);
//			context.write(key, result);
			
			// sum为0，总得单词数加1，统计所有单词的种类
			if (sum == 0) {
				context.getCounter(WordsNature.TOTALWORDS).increment(1);
			} else {// sum不为0时，通过key的长度来判断，
				result.set(sum);
				String[] temp = key.toString().split("\t");
				if (temp.length == 2) { // 用tab分隔类别和单词
					
					context.write(key, result);
					// mos.write(new Text(temp[1]), result, temp[0]);
				}
				
				else { // 类别中单词总数	
					mos.write(key, result, "_wordsInClass" + "\\" + "wordsInClass");
				}
			}
		}

		@Override
		protected void cleanup(Reducer<Text, IntWritable, Text, IntWritable>.Context context)
				throws IOException, InterruptedException {

			mos.close();
		}

		@Override
		protected void setup(Reducer<Text, IntWritable, Text, IntWritable>.Context context)
				throws IOException, InterruptedException {

			mos = new MultipleOutputs<Text, IntWritable>(context);
		}

	}

	public static int run(Configuration conf) throws Exception {
		Job job = new Job(conf, "word count");

		job.setJarByClass(WordCount.class);

		job.setInputFormatClass(MyInputFormat.class);

		job.setMapperClass(WordCount.WordCountMapper.class);
		job.setReducerClass(WordCount.WordCountReducer.class);
//		System.out.println("-1--job-------");
		// 过滤掉文本数少于FILENUMBER的类别

		String input = conf.get("input");
		
		List<Path> inputPaths = MyUtils.getSecondDir(conf, input);
		for (Path path : inputPaths) {
//			System.out.println("-3----path = " + path.toString());
			MyInputFormat.addInputPath(job, path);
		}

		String wordsOutput = conf.get("wordsOutput");
		FileOutputFormat.setOutputPath(job, new Path(wordsOutput));

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);

		int exitCode = job.waitForCompletion(true) ? 0 : 1;

		// 调用计数器
		Counters counters = job.getCounters();
		Counter c1 = counters.findCounter(WordsNature.TOTALWORDS);
		System.out.println("------------>>>>: " + c1.getDisplayName() + ":" + c1.getName() + ": " + c1.getValue());
		
		
		// 将单词种类数写入文件中，只是为了看看数据
		Path totalWordsPath = new Path("hdfs://192.168.190.128:9000/user/hadoop/output/totalwords.txt");
		FileSystem fs = totalWordsPath.getFileSystem(conf);
		FSDataOutputStream outputStream = fs.create(totalWordsPath);
		outputStream.writeBytes(c1.getDisplayName() + ":" + c1.getValue());
		IOUtils.closeStream(outputStream);

		// 直接单词总种类数写到configuration中，需要直接读取不用读文件
		conf.setInt("TOTALWORDS", (int)c1.getValue());

		return exitCode;

	}

}
