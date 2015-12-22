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

	// ������LOG��ӡ������Ϣ
	private static final Log LOG = LogFactory.getLog(FileInputFormat.class);
	public static int total = 0; // �������е��ʵ��������
	private static MultipleOutputs<Text, DoubleWritable> mos;

	public static int run(Configuration conf) throws Exception {

		// �ȶ�ȡ�����������
		total = conf.getInt("TOTALWORDS", 0);
		LOG.info("------>total = " + total);// ��ȡ�������ĵ�������������ֻΪ�ڿ���̨��ʾ��������

		System.out.println("total ==== " + total);// Ҳ�����ڿ���̨��ʾ��ӡ��Ϣ

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
		
		private static Map<String, Integer> fileMap = new HashMap<String, Integer>();// ��������е�������
		
		protected void setup(Mapper<LongWritable, Text, Text, DoubleWritable>.Context context)
				throws IOException, InterruptedException {
			mos = new MultipleOutputs<Text, DoubleWritable>(context);//�ļ��Ķ��ļ������
			
			Configuration conf = context.getConfiguration();
			String filePath = conf.get("wordsInClassPath");//��ȡ����е����������ļ�·��
			FileSystem fs = FileSystem.get(URI.create(filePath), conf);//��ȡ�ļ�ϵͳ����
			FSDataInputStream inputStream = fs.open(new Path(filePath));//�����ļ�����������
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
			int tot = conf.getInt("TOTALWORDS", 0);// map����Ҫ���»�ȡ������������

			// System.out.println("total = " + total);//total = 0�޷�ʹ�ã�ֻ��ͨ��conf��ֵ
			// System.out.println("tot = " + tot);//����������userlog�е�stdout����ʾ

			// ����ĸ�ʽ���£�
			// ALB weekend 1
			// ALB weeks 3
			Map<String, Map<String, Integer>> baseMap = new HashMap<String, Map<String, Integer>>(); // ����������ݣ�<����,<���ʣ���Ƶ>>
			// Map<String, Map<String, Double>> priorMap = new HashMap<String,
			// Map<String, Double>>(); // ����ÿ�����ʳ��ֵĸ���

			String[] temp = value.toString().split("\t");
			// �Ƚ����ݴ浽baseMap��
			if (temp.length == 3) {
				// �ļ����������
				if (baseMap.containsKey(temp[0])) {
					baseMap.get(temp[0]).put(temp[1], Integer.parseInt(temp[2]));
				} else {
					Map<String, Integer> oneMap = new HashMap<String, Integer>();
					oneMap.put(temp[1], Integer.parseInt(temp[2]));
					baseMap.put(temp[0], oneMap);
				}
			} // ��ȡ������ϣ�ȫ��������baseMap��

			int allWordsInClass = 0;

			for (Map.Entry<String, Map<String, Integer>> entries : baseMap.entrySet()) { // �������
				allWordsInClass = fileMap.get(entries.getKey());
				for (Map.Entry<String, Integer> entry : entries.getValue().entrySet()) { // ��������еĵ��ʴ�Ƶ�����
					double p = (entry.getValue() + 1.0) / (allWordsInClass + tot);// �������0��ÿ�����ʶ���1

					className.set(entries.getKey() + "\t" + entry.getKey());
					number.set(p);
//					LOG.info("------>p = " + p);//������������userlog�е�syslog�в�����ʾ����
					mos.write(new Text(entry.getKey()), number, entries.getKey());// ��ÿ�����ĵ��ʴ�Ƶ�ֱ�д���������������ļ���
					// context.write(className, number);
				}
			}
		}
	
		protected void cleanup(Mapper<LongWritable, Text, Text, DoubleWritable>.Context context)
				throws IOException, InterruptedException {
			// ����������в����ڵ��ʵĸ��ʣ�ÿ�������һ������
			Configuration conf = context.getConfiguration();
			int tot = conf.getInt("TOTALWORDS",0);//��ȡ��������е�������
			for (Map.Entry<String, Integer> entry : fileMap.entrySet()) { // ������������е��ʵ��������õ����е����
				double notFind = (1.0) / (entry.getValue() + tot);
				number.set(notFind);
				mos.write(new Text(entry.getKey()), number, "_notFound" + "\\" + "notFound");//�ļ���ǰ��_�����ú�����mapreduce���Ը��ļ���

			}
			mos.close();
		}
	}

	// Reducer
	static class ProbReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
		//ʲôҲ������
		
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
			context.write(key, result);// ֻ�����򵥵Ĺ鲢
		}*/
	}
}
