abstract public class Konto implements KontoInterface, Comparable
{
    protected String inhaber;
    protected double saldo;

    public Konto(String inhaber)
    {
        this.inhaber = inhaber;
        saldo = 0;
    }

	
    public double getSaldo()
	{
		return saldo;
	}


    public void einzahlen(double betrag)
    {
        if(betrag<0) {
            System.out.println("Es koennen nur positive Betraege eingezahlt werden.");
            return;
        }

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
            System.out.println("Es können nur positive Betraege abgehoben werden.");
            return;
        }

        if(deckungPruefen(betrag) == false) {
            System.out.println("Konto nicht ausreichend gedeckt.");
            return;
        }

        saldo -= betrag;
        System.out.println("Es wurden " + betrag + " Euro ausgezahlt.");
        System.out.println("  Neuer Saldo von " + inhaber + ": " + saldo);
    }

    public String toString()
    {
        return "Saldo von " + inhaber + ": " + saldo;
    }

	public int compareTo(Object o){
		if(!(o instanceof Konto)) {
			return 0;
		}

		Konto k = (Konto)o;
		if(saldo > k.saldo)
			return 1;
		else if(saldo < k.saldo)
			return -1;
		else
			return 0;

	}
}
