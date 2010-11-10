abstract public class Konto implements KontoInterface, Comparable
{
    protected String inhaber;
    protected double saldo;

    public Konto(String inhaber)
    {
        this.inhaber = inhaber;
        saldo = 0;
    }

    public void einzahlen(double betrag)
    {
        if(betrag<0) {
            System.out.println("Es koennen nur positive Betraege eingezahlt werden.");
            return;
        }

        // Test erfolgreich, Einzahlung durchfuehren.
        saldo += betrag;
        System.out.println("Es wurden " + betrag + " Euro eingezahlt.");
        System.out.println("  Neuer Saldo von " + inhaber + ": " + saldo);
    }
    
    public boolean deckungPruefen(double betrag)
    {
        if(saldo - betrag < 0)
            return false;
        else 
            return true;
    }

    public void abheben(double betrag)
    {
        if(betrag<0) {
            System.out.println("Es koennen nur positive Betraege ausgezahlt werden.");
            return;
        }

        if(deckungPruefen(betrag) == false) {
            System.out.println("Konto nicht ausreichend gedeckt.");
            return;
        }

        // Tests erfolgreich, Abhebung durchfuehren.
        saldo -= betrag;
        System.out.println("Es wurden " + betrag + " Euro ausgezahlt.");
        System.out.println("  Neuer Saldo von " + inhaber + ": " + saldo);
    }
	
	public double getSaldo()
	{
		return saldo;
	}
	
	public int compareTo(Konto tmp)
	{
		if (this.saldo > tmp.saldo) return 1;
		if (this.saldo < tmp.saldo) return -1;
		else return 0;
	}

    public String toString()
    {
        return "Saldo von " + inhaber + ": " + saldo;
    }
	    
    public static void main(String args[])
    {
        Girokonto g = new Girokonto("Peter", 1000);
        g.einzahlen(500);
        g.abheben(700);
		
        Tagesgeldkonto t = new Tagesgeldkonto("Hans");
        g.ueberweisen(t, 100);
        Tagesgeldkonto.setzeZinssatz(5);
        t.zinsgutschrift();
    }
}
