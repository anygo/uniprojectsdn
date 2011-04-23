
public class LinkedList implements List {
	
	protected static class ListNode implements List.Node {
		public int data;
		public ListNode next;
		public int getData() {
			return data;
		}

		public ListNode(int a) {
			data = a;
		}
	}

	
	private ListNode root;
	private int size;

	public LinkedList() {
		root = null;
		size = 0;
	}

	public void add(int a) {
		if (root == null) {
			root = new ListNode(a);
			size++;
			return;
		} else {
			ListNode tmp = new ListNode(a);
			tmp.next = root;
			root = tmp;
			size++;
		}
	}

	public void insert(Node n, int a) {
		if (root == null) {
			System.out.println("Fehler: Liste leer (insert(Node n, int a))");
			return;
		}
		ListNode tmp = (ListNode)n;
		ListNode current = root;
		while (current != tmp) {
			current = (ListNode)getNext(current);
			if (current == null) {
				System.out.println("Fehler: Knoten nicht gefunden (insert(Node n, int a))");
				return;
			}
		}
		ListNode newNode = new ListNode(a);
		newNode.next = (ListNode)getNext(current);
		tmp.next = newNode;
		size++;
	}

	public Node find(int a) {
		if (root == null) {
			System.out.println("Fehler: Liste leer (find(inta a))");
		}
		ListNode current = (ListNode)root;
		while (current.data != a) {
			current = (ListNode)getNext(current);
			if (current == null) {
				System.out.println("Fehler: Knoten nicht gefunden (find(int a))");
				return null;
			}
		}
		return current;
	}

	public void remove(int a) {
		ListNode toRemove = (ListNode)find(a);
		if (toRemove == root) {
			remove(root);
			add(a);
			size--;
			return;
		}
		if (toRemove == null) {
			System.out.println("Fehler: null von find (remove(int a))");
			return;
		}
		ListNode previous = (ListNode)getPrevious(toRemove);
		previous.next = (ListNode)getNext(toRemove);
		size--;
	}

	public void remove(Node e) {
		ListNode toRemove = (ListNode)e;
		if (root == toRemove) {
			root = root.next;
			size--;
			return;
		}
		ListNode previous = (ListNode)getPrevious(e);
		previous.next = toRemove.next;
		size--;
	}

	public Node getFirst() {
		return (ListNode)root;
	}

	public Node getNext(Node e) {
		ListNode tmp = (ListNode)e;
		return tmp.next;
	}

	public Node getLast() {
		if (root == null) return null;
		ListNode tmp = root;
		while (tmp.next != null) tmp = tmp.next;
		return tmp;
	}

	public Node getPrevious(Node e) {
		ListNode tmp = (ListNode)e;
		if (root == null) return null;
		if (root == tmp) return null;
		ListNode current = root;
		while ((ListNode)current.next != null && (ListNode)current.next != tmp) current = (ListNode)current.next;
		return current;
	}

	public int size() {
		return size;
	}

	public String toString() {
		if (root == null) return "Liste leer";
		ListNode current = (ListNode)root;
		String toReturn = new String();
		while (current.next != null) {
			toReturn = toReturn + current.data + " ";
			current = current.next;
		}
		toReturn += current.data;
		return toReturn;
	}

	
	public static void main(String[] args) {
		LinkedList list = new LinkedList();
		System.out.println(list);
		list.add(2);
		list.add(55);
		list.add(213);
		list.add(123);
		list.add(222);
		list.remove(222);

		System.out.println(list);
	}

}
