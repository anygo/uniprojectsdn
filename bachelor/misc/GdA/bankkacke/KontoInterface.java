interface KontoInterface
{
    public void einzahlen(double betrag);
    public void abheben(double betrag);
    public double getSaldo();
    
    public boolean deckungPruefen(double betrag);

	//Gibt Informationen ueber das Konto als Zeichenkette zurueck (z.B.: Name und Saldo)
	public String toString();
}
