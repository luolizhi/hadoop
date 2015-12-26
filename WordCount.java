package org.lukey.hadoop.bayes.trainning;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
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
	private static final Log LOG = LogFactory.getLog(FileInputFormat.class);//试着用LOG打印调试信息	
	private static MultipleOutputs<Text, IntWritable> mos;//将结果输出到不同的文件名中	
	static enum WordsNature {//使用计数器，前面两个没有用好
		CLSASS_NUMBER, CLASS_WORDS, TOTALWORDS
	}
	static class WordCountMapper extends Mapper<Text, Text, Text, IntWritable> {
		private final static IntWritable one = new IntWritable(1);
		private final static IntWritable zero = new IntWritable(0);
		private Text countryName = new Text();	
		@Override
		protected void map(Text key, Text value, Mapper<Text, Text, Text, IntWritable>.Context context)
				throws IOException, InterruptedException {
			LOG.info("--map0--- " + key + "---------");	//在logs/userlogs中的syslog输出
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
		}
	}

	static class WordCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {		
		IntWritable result = new IntWritable();// result 表示每个类别中每个单词的个数
		Map<String, List<String>> classMap = new HashMap<String, List<String>>();//
		Map<String, List<String>> fileMap = new HashMap<String, List<String>>();//

		@Override
		protected void reduce(Text key, Iterable<IntWritable> values,
				Reducer<Text, IntWritable, Text, IntWritable>.Context context)
						throws IOException, InterruptedException {
			System.out.println("reduce reduce");
			//LOG.info("---reduce-- " + key);
			int sum = 0;
			for (IntWritable value : values) {
				sum += value.get();
			}						
			if (sum == 0) {// sum为0，总得单词数加1，统计所有单词的种类
				context.getCounter(WordsNature.TOTALWORDS).increment(1);
			} else {// sum不为0时，通过key的长度来判断，
				result.set(sum);
				String[] temp = key.toString().split("\t");
				if (temp.length == 2) { // 用tab分隔类别和单词				
					context.write(key, result);
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
		
		job.setInputFormatClass(MyInputFormat.class);//自己定义的InputFormat
		job.setMapperClass(WordCount.WordCountMapper.class);
		job.setReducerClass(WordCount.WordCountReducer.class);
		
		String input = conf.get("input");// 过滤掉文本数少于FILENUMBER的类别		
		List<Path> inputPaths = MyUtils.getSecondDir(conf, input);
		for (Path path : inputPaths) {
			MyInputFormat.addInputPath(job, path);
		}

		String wordsOutput = conf.get("wordsOutput");
		FileOutputFormat.setOutputPath(job, new Path(wordsOutput));

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);

		int exitCode = job.waitForCompletion(true) ? 0 : 1;	
		Counters counters = job.getCounters();// 调用计数器
		Counter c1 = counters.findCounter(WordsNature.TOTALWORDS);
		//System.out.println("------------>>>>: " + c1.getDisplayName() + ":" + c1.getName() + ": " + c1.getValue());			
		conf.setInt("TOTALWORDS", (int)c1.getValue());// 直接单词总种类数写到configuration中，需要直接读取不用读文件
		return exitCode;
	}
}
