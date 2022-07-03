import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.hadoop.util.GenericOptionsParser;

public class InvertedIndex {
	public static class InvertedIndexMapper extends Mapper<Object, Text, Text, IntWritable> {
		private Set<String> stopwords;
		private Path cacheFiles;

		public void setup(Context context) throws IOException, InterruptedException {
			stopwords = new TreeSet<String>();
			Configuration conf = context.getConfiguration();
			FileSystem fs = FileSystem.get(conf);
			URI[] files = context.getCacheFiles();// 获取所有的缓存的停词文件的URI
			for (int i = 0; i < files.length; ++i) {
				cacheFiles = new Path(files[i]);
				String line;
				BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(cacheFiles)));// 读取该文件
				while ((line = br.readLine()) != null) {
					StringTokenizer itr = new StringTokenizer(line);
					while (itr.hasMoreTokens())
						stopwords.add(itr.nextToken());// 将停词存入set
				}
				br.close();
			}
		}

		/**
		 * map(): 对输入的Text切分为多个word 输入：key:当前行偏移位置, value:当前行内容 输出：key:word,filename
		 * value:1
		 */
		protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			FileSplit fileSplit = (FileSplit) context.getInputSplit();
			String fileName = fileSplit.getPath().getName();// .toLowerCase();//获取文件名，转换为小写
			String line = value.toString().toLowerCase();// 将行内容全部转为小写字母
			// 只保留数字和字母
			String new_line = "";
			for (int i = 0; i < line.length(); i++) {//将数字和字母以外的字符换为空格
				if ((line.charAt(i) >= '0' && line.charAt(i) <= '9')
						|| (line.charAt(i) >= 'a' && line.charAt(i) <= 'z'))
					new_line += line.charAt(i);
				else
					new_line += " ";// 其他字符保存为空格
			}
			line = new_line.trim();// 去掉开头和结尾的空格
			StringTokenizer strToken = new StringTokenizer(line);// 按照空格拆分
			while (strToken.hasMoreTokens()) {
				String str = strToken.nextToken();
				if (!stopwords.contains(str))// 不在停词表则输出key-value对
					context.write(new Text(str + "," + fileName), new IntWritable(1));//<key:<word,docid>,value:1>
			}
		}
	}

	public static class SumCombiner extends Reducer<Text, IntWritable, Text, IntWritable> {
		public void reduce(Text key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val : values)
				sum++;
			context.write(key, new IntWritable(sum));//将相同的 word,docid 合并，减少向reduce节点传输的数据量
		}
	}

	public static class NewPartitioner extends HashPartitioner<Text, IntWritable> {
		public int getPartition(Text key, IntWritable value, int numReduceTasks) {
			String term = key.toString().split(",")[0];//获取 word,docid 的word
			return super.getPartition(new Text(term), value, numReduceTasks);//让父类使用word进行hash来决定分给哪个节点
		}
	}

	public static class InvertedIndexReducer extends Reducer<Text, IntWritable, Text, Text> {
		private String lastfile = null;// 存储上一个filename
		private String lastword = null;// 存储上一个word
		private String str = "";// 存储要输出的value内容
		private int count = 0;//该词在一个文件里的出现次数
		private int totalcount = 0;//该词在全部文件里的出现次数

		/**
		 * 利用每个Reducer接收到的键值对中，word是排好序的 将word，filename拆分开，将filename与累加和拼到一起，存在str中
		 * 每次比较当前的word和上一次的word是否相同，若相同则将filename和累加和附加到str中 否则输出：key:word，value:str
		 * 并将新的word作为key继续
		 * 输入：key:(word,filename), value:[num1,num2,...] 输出：key:word, value:filename1:sum1;filename1:sum1;...
		 */
		protected void reduce(Text key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			String[] tokens = key.toString().split(",");// 将word和filename存在tokens数组中
			if (lastword == null)
				lastword = tokens[0];
			if (lastfile == null)
				lastfile = tokens[1];
			if (!tokens[0].equals(lastword)) {// 此次word与上次不一样，则将上次的word进行处理并输出
				str += "<" + lastfile + "," + count + ">;<total," + totalcount + ">.";
				context.write(new Text(lastword), new Text(str));// value部分拼接后输出
				lastword = tokens[0];// 更新word
				lastfile = tokens[1];// 更新filename
				count = 0;//重置为0
				str = "";
				for (IntWritable val : values)// 累加相同word和filename中出现次数
					count += val.get();// 转为int
				totalcount = count;
				return;
			}
			if (!tokens[1].equals(lastfile)) {// 新的文档
				str += "<" + lastfile + "," + count + ">;";
				lastfile = tokens[1];// 更新文档名
				count = 0;// 重设count值
				for (IntWritable value : values)// 计数
					count += value.get();// 转为int
				totalcount += count;
				return;
			}
			// 其他情况，只计算总数即可
			for (IntWritable val : values) {
				count += val.get();
				totalcount += val.get();
			}
		}

		/**
		 * 上述reduce()只会在遇到新word时，处理并输出前一个word，故对于最后一个word还需要额外的处理
		 * 重载cleanup()，处理最后一个word并输出
		 */
		public void cleanup(Context context) throws IOException, InterruptedException {
			str += "<" + lastfile + "," + count + ">;<total," + totalcount + ">.";
			context.write(new Text(lastword), new Text(str));
			super.cleanup(context);
		}
	}

	public static void main(String[] args) throws Exception {
		args = new String[] { "hdfs://localhost:9000/ex2/input", "hdfs://localhost:9000/ex2/output" };
		//args = new String[] { "hdfs://10.102.0.198:9000/input", "hdfs://10.102.0.198:9000/user/bigdata_202000202092/output" };
		Configuration conf = new Configuration();
		conf.set("fs.defaultFS", "hdfs://localhost:9000");
		//conf.set("fs.defaultFS", "hdfs://10.102.0.198:9000");
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		if (otherArgs.length != 2) {
			System.err.println("ERROR");
			System.exit(2);
		}
		Path path = new Path(args[1]);
		FileSystem fileSystem = path.getFileSystem(conf);
		if (fileSystem.exists(new Path(args[1])))// 如果输出目录存在就删除它
			fileSystem.delete(new Path(args[1]), true);
		Job job = Job.getInstance(conf, "InvertedIndex");
		job.addCacheFile(new URI("hdfs://localhost:9000/ex2/stop_words/stop_words_eng.txt"));
		//job.addCacheFile(new URI("hdfs://10.102.0.198:9000/stop_words/stop_words_eng.txt"));
		job.setJarByClass(InvertedIndex.class);
		job.setMapperClass(InvertedIndexMapper.class);
		job.setCombinerClass(SumCombiner.class);
		job.setReducerClass(InvertedIndexReducer.class);
		job.setPartitionerClass(NewPartitioner.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);
		FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}
