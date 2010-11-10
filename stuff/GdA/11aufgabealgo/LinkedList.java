public class LinkedList implements List {
    
	//Stellt einen Iterator fuer eine verkettetet Liste zur verfuegung
	private class LinkedListIterator implements Iterator {
	
		public Node whereAmI;
		public LinkedList iterator_list;

		//Erzeugt einen neuen Iterator fuer die verkettete Liste
		public LinkedListIterator(LinkedList ll){
			
			whereAmI = ll.head;
			iterator_list = ll;
		}

		//Gibt true zurueck, wenn es noch ein weiteres Element in der Liste gibt
		public boolean hasNext(){
			
				return false;
			
			//if (whereAmI != iterator_list.getLast() && whereAmI != null)
		//		return true;
		//	else
		//		return false;
		}

		//Gibt das naechste Element der Liste zurueck
		public Node next(){
			
			return iterator_list.getNext(whereAmI);
		}

		//Setzt den iterator auf den Anfang zurueck
		public void reset(){
			
			whereAmI = iterator_list.head;
		}
	}

	// Erzeugt einen neuen Iterator und gibt eine Referenz zurueck
	public Iterator getIterator(){
		
		LinkedListIterator it = new LinkedListIterator(this);
		return it;
	}

	// **********************************************************************
	// Einige einfache Methode zum testen des Iterators
	// **********************************************************************
	public static void iterateThroughList(Iterator i){
		System.out.print("Iterator Ausgabe: ");
		if (i!=null) {
			// zuerst muss der iterator zuruckgesetzt werden
			i.reset();

			// solange Elemente in der Liste zur verfuegung stehen widerholen...
			while (i.hasNext()){
				//das naechste element laden und den iterator eine Position weiter verschieben
				List.Node val = i.next();
				System.out.print(val.getData() + " ");
			}
		}

		System.out.println("\n");
	}

	public static void main(String[] args){
		LinkedList list = new LinkedList();

		list.add(10);
		list.add(90);
		list.add(42);
		list.add(19);

		System.out.println("Normale Ausgabe: " + list);
		iterateThroughList(list.getIterator());
	}




// **********************************************************************
// Ab hier bitte nichts mehr aendern !!!
// **********************************************************************
    protected static class ListNode implements Node {
        public int data;
        public ListNode next;
        
        public int getData() { 
            return data; 
        }
    }
   
    protected ListNode head = null;
    
    public void add(int a) {
        ListNode l = new ListNode();
        l.data = a;
        l.next = head;
        head = l;
    }

    public void insert(Node e, int a) {
        ListNode l = new ListNode();
        ListNode n = ((ListNode)e);
        l.data = a;
        l.next = n.next;
        n.next = l;
    }

    
    public void remove(int a) {
        ListNode toBeRemoved = (ListNode)find(a);
        if(toBeRemoved != null)
            remove(toBeRemoved);
    }
    
    public Node find(int a) {
        ListNode current = head;
        while(current != null) {
            if(current.data == a)
                return current;
            current = current.next;
        }
        return null;
    };
        
    public void remove(Node e) {
        ListNode n = (ListNode)e;

        if(head == n)
            head = n.next;
        else {
            ListNode prev = (ListNode)getPrevious(n);
            prev.next = n.next;
        }
    }
    
    public Node getFirst() {
        return head;
    }
    
    public Node getNext(Node e) {
        return ((ListNode)e).next;
    }

    public Node getLast() {
        if(head == null)
            return null;

        ListNode current = head;
        while(current.next != null)
            current = current.next;
        return current;
    }
        
    public Node getPrevious(Node e) {
        if(head == e || head == null)
            return null;
        
        ListNode current = head;
        while(current.next != null) {
            if(current.next == e)
                return current;
            current = current.next;
        }

        return null;
    }
    
    public int size() {
        if(head == null)
            return 0;

        int c = 1;
        ListNode current = head;
        while(current.next != null) {
            current = current.next;
            c++;
        }
        return c;
    }
    
    public String toString() {
        ListNode current = head;
        String s = "";
        while(current != null) {
            s += current.data + " ";
            current = current.next;
        }
        return s;
    }
}

