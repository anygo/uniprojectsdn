package agreement;

/**
 * Transform a series of real valued measures into fixed categories.
 * @param data Array containing real valued measures
 * @param nc number of classes
 * @author sikoried
 */
public class Categorizer 
{
	/**
	 * Categorize into given classes using lower bounds.
	 * @param data real valued measures
	 * @param lowerBounds lower bounds to categorize
	 * @return new allocated array containing categorized data
	 */
	public static double [] categorize(double [] data, double [] lowerBounds)
	{
		double [] cat = new double [data.length];
		for (int i = 0; i < data.length; ++i)
		{
			for (int c = 0; c < lowerBounds.length; c++)
				if (data[i] > lowerBounds[c])
					cat[i] = lowerBounds.length - c + 1;
		}		
		return cat;
	}
	/**
	 * Categorize into nc classes
	 * @param data real valued measures
	 * @param nc number of classes
	 * @return new allocated array containing categorized data
	 */
	public static double [] categorizeEquidistant(double [] data, int nc)
	{
		double [] trans = new double[nc-1];
		
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		
		for (double d : data)
		{
			if (d < min)
				min = d;
			if (d > max)
				max = d;
		}
		
		double range = max - min;
		
		for (int i = 0; i < trans.length; ++i)
			trans[i] = min + (i+1)*(range/(1.+nc));
		
		return categorize(data, trans);
	}
	public static double [] categorizeOptimal(double[][] categorized, double [] realvalued, AgreementMeasure measure, int nc)
		throws Exception
	{
		double [] trans = new double[nc -1];

		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		
		for (double d : realvalued)
		{
			if (d < min)
				min = d;
			if (d > max)
				max = d;
		}
		
		double range = max - min;
		
		for (int i = 0; i < trans.length; ++i)
			trans[i] = min + (i+1)*(range/(1.+nc));
		
		double kappaNew = measure.agreement(Utils.addRater(categorized, categorize(realvalued, trans)));
		double kappaOld = kappaNew;
		int maxi = 10;
		for (int i = 0; i < maxi; i++)
		{
			for (int j= 0; j< trans.length;j++) 
			{
				for (int k = 0; k < 3;k++)
				{
					boolean increase = true;
					if (j != trans.length -1)
						if (trans[j] + (maxi-i) > trans[j+1]) increase = false;  // Cannot increase boundary
					if (increase) trans[j] += (maxi-i);
					kappaNew = measure.agreement(Utils.addRater(categorized, categorize(realvalued, trans)));
					if (! (kappaNew > kappaOld)) 
					{
						boolean decrease = true;
						int add = (increase) ? (2 * (maxi-i)): (maxi -i);
						if (j != 0) if (trans[j] - add < trans[j-1]) decrease = false;
						if (decrease) trans[j] -= add;
						else if (increase) trans[j] -= maxi -i;
						kappaNew = measure.agreement(Utils.addRater(categorized, categorize(realvalued, trans)));
						if ((kappaNew > kappaOld)) 
							kappaOld = kappaNew;
						else if (decrease) {
							if (increase) trans[j] += maxi -i;
							else trans[j] += add;
						}
					} 
					else 
						kappaOld = kappaNew;
				}
			}
		}
		System.out.println("found optimal categorization: [ ");
		for (int i = 0; i < trans.length;i++) 
			System.out.print(trans[i] + " ");
		System.out.println("]");
		
		return categorize(realvalued, trans);
	}
}
