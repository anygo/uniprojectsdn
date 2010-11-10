public class Bank implements BankVerwaltungOnline
{
	Konto[] konten;

	Bank()
	{
		konten = new Konto[0];
	}

	public void addKonto(Konto k)
	{
		//falls Konto bereits vorhanden...
		if (haveKonto(k) == true)
		{
			System.out.println("Das Konto ist bereits vorhanden.");
			return;
		}

		Konto[] tmp = new Konto[konten.length + 1];
		for (int i = 0; i < konten.length; i++)
			{
			tmp[i] = konten[i];
			}
		tmp[konten.length] = k;
		konten = tmp;
	}

	public void removeKonto(Konto k)
	{
		//ueberpruefen, ob Konto vorhanden ist
		if (haveKonto(k) == false)
		{
			System.out.println("Das Konto ist nicht vorhanden und kann deshalb nicht geloescht werden.");
			return;
		}

		Konto[] tmp = new Konto[konten.length - 1];
		int j = 0;
		for (int i = 0; i < konten.length; i++)
		{
			if (konten[i] != k)
			{
				tmp[j] = konten[i];
				j++;
			}
		}
		konten = tmp;
	}

	public boolean haveKonto(Konto k)
	{
		boolean haveKontoBoolean = false;
		for (int i = 0; i < konten.length; i++)
		{
			if (k == konten[i]) 
			{
				haveKontoBoolean = true;
			}
		}
		return haveKontoBoolean;
	}

	public int verwalteteKonten()
	{
		return konten.length;
	}

	public String toString()
	{
		String s = new String();
		for (int i = 0; i < konten.length; i++)
		{
			s += (konten[i].inhaber);
			if (i != konten.length - 1)
			{
				s += (", ");
			}
		}
		return s;
	}

	public Konto hoechsterSaldo()
	{
		return null;		
	}

}
