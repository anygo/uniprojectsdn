class Maximum
{
	public static Comparable max(Comparable[] data)
	{
		int maxindex=0;
		for (int i = 0; i < data.length; i++)
		{
			if (data[i].compareTo(data[maxindex]) == 1) 
			{
				maxindex = i;
			}
		}
		return data[maxindex];
	}

	public static void main(String[] args)
	{
		Integer[] test = {6, 3, 4, 1, 3, 7, 1, 2};
		System.out.println(max(test));
	}
}
