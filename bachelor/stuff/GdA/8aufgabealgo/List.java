public interface List {
    // Das Interface fuer Listeneintraege.  Da die Listenstruktur von aussen
    // nicht veraendert werden soll, exponieren wir nur dieses Interface. 
    public interface Node {
        public int getData();
    }
    
    // Erstellt einen neuen Listeneintrag fuer das Element a und fuegt 
    // ihn am Kopf der Liste hinzu.
    public void add(int a);

    // Erstellt einen neuen Listenknoten f"ur und f"ugt ihn nach n in der Liste
    // ein.
    public void insert(Node n, int a);

    // Findet das erste Vorkommen einer Node in der Liste und gibt eine
    // Referenz auf den jeweiligen Listeneintrag zurueck (oder null,
    // wenn es nicht enthalten ist).
    public Node find(int a); 

    // Findet das erste Vorkommen eines Werts und entfernt den
    // entsprechenden Listeneintrag aus der Liste.
    public void remove(int a);

    // Entfernt den Listeneintrag aus der Liste.
    public void remove(Node e);
    
    // Gibt den ersten Eintrag der Liste zurueck.
    public Node getFirst();

    // Gibt den Listeneintrag zurueck, der Nachfolger von e ist.
    public Node getNext(Node e);

    // Gibt den letzten Eintrag der Liste zurueck.
    public Node getLast();
    
    // Gibt den Listeneintrag zurueck, dessen Nachfolger e ist.
    public Node getPrevious(Node e);
    
    // Gibt die Anzahl der Listeneintraege zurueck.
    public int size(); 
}

