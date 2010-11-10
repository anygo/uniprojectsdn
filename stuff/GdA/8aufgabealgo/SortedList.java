
public class SortedList extends LinkedList {
	
	public void add(int a) {
		ListNode neuerKnoten = new ListNode();
		neuerKnoten.data = a;
		ListNode tmp = listenkopf;
		
		if (listenkopf == null) {
			listenkopf = neuerKnoten;
			return;
		}

		if (listenkopf.next == null) {
			if (listenkopf.data <= a) { 
				listenkopf.next = neuerKnoten;
			} else {
				neuerKnoten.next = listenkopf;
				listenkopf = neuerKnoten;
			}
			return;
		}

		if (a <= listenkopf.data) {
			neuerKnoten.next = listenkopf;
			listenkopf = neuerKnoten;
			return;
		}

		while (tmp.next != null) {
			if (a <= tmp.next.data) {
				neuerKnoten.next = tmp.next;
				tmp.next = neuerKnoten;
				return;
			}
			tmp = tmp.next;
		}
		tmp.next = neuerKnoten;
	}
}
