public class Maximum {
    public static Comparable max(Comparable[] data) {
        if(data == null || data.length == 0)
            return null;

        Comparable m = data[0];
        for(Comparable c : data) {
            if(m.compareTo(c) < 0) {
                m = c;
            }
        }
        
        return m;
    }

    public static void main(String[] args)
    {
        Integer[] a = new Integer[] {3, 6, 2, 8, 5};
        System.out.println(Maximum.max(a));
    }
}
