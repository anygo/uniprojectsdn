package vsue.totalorder;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import org.jgroups.Address;
import org.jgroups.Global;
import org.jgroups.util.Streamable;
import org.jgroups.util.Util;

public class VSMsgID implements Streamable
{
	//-- Needed for serialization
	public VSMsgID()
	{	
	}
	
	
	public VSMsgID(Address addr, long no)
	{
		if( addr==null )
			throw new NullPointerException( "addr" );
		
		m_addr  = addr;
		m_no 	= no;
	}
	
	
	public Address getAddress()
	{
		return m_addr;
	}
	
	
	public long getNumber()
	{
		return m_no;
	}
	
	
	@Override
	public void readFrom(DataInputStream in)
					throws IOException, IllegalAccessException, InstantiationException
	{
		m_addr  = Util.readAddress( in );
		m_no = in.readLong();
	}


	@Override
	public void writeTo(DataOutputStream out) throws IOException
	{
		Util.writeAddress( m_addr, out );
		out.writeLong( m_no );
	}
	
	
	public int size()
	{
		return Util.size( m_addr ) + Global.LONG_SIZE;
	}

	
	@Override
	public boolean equals(Object obj)
	{
		if( obj==null || !(obj instanceof VSMsgID) )
			return false;
		else
		{
			VSMsgID msgid = (VSMsgID) obj;
			
			return m_addr.equals( msgid.m_addr ) && m_no==msgid.m_no;
		}
	}
	
	
	@Override
	public int hashCode()
	{
		return m_addr.hashCode() ^ ((Long) m_no).hashCode();
	}


	@Override
	public String toString()
	{
		return m_addr + "#" + m_no;
	}


	private Address		m_addr;
	private long		m_no;
}
