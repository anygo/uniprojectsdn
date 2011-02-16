package mw.cliquefind;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class FindCliqueTool extends Configured implements Tool {

	@Override
	public int run(String[] arg0) throws Exception {
		FindTrussIteration fti = new FindTrussIteration();
		
		System.out.println("n: " + getConf().get("n"));
		System.out.println("Input: " + getConf().get("input"));
		System.out.println("Output: " + getConf().get("output")); 
		
		fti.run(getConf());

		return 0;
	}
	
	public static void main(String args[]) throws Exception {
		int res = ToolRunner.run(new Configuration(), new FindCliqueTool(), args);
		
		System.exit(res);
	}
}
