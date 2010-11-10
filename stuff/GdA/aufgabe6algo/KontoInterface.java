interface KontoInterface
{
    void einzahlen(double betrag);
    void abheben(double betrag);
    double getSaldo();
    
    boolean deckungPruefen(double betrag);

	String toString();
}
