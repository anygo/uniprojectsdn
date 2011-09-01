package vsue.totalorder;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import org.jgroups.Global;
import org.jgroups.Header;
import org.jgroups.ViewId;

public class VSTotalOrderHeader extends Header
{
	//-- Needed for serialization
	public VSTotalOrderHeader()
	{
	}
	
	
	public static VSTotalOrderHeader createRerouting(VSMsgID msgid)
	{
		return new VSTotalOrderHeader( VSTotalOrderMsgType.REROUTING, msgid, null, -1L );
	}
	
	
	public static VSTotalOrderHeader createMulticast(VSMsgID msgid)
	{
		return createMulticast( msgid, null, -1 );
	}
	
	
	public static VSTotalOrderHeader createMulticast(VSMsgID msgid, ViewId viewid, long order)
	{
		return new VSTotalOrderHeader( VSTotalOrderMsgType.MULTICAST, msgid, viewid, order );
	}
	
	
	public VSTotalOrderHeader createAck()
	{
		if( m_msgtype!=VSTotalOrderMsgType.MULTICAST )
			throw new IllegalStateException( "Only multicasts can be acknowledged." );
		
		return new VSTotalOrderHeader( VSTotalOrderMsgType.ACK, m_msgid, m_viewid, m_order );
	}
	
	
	public VSTotalOrderMsgType getMsgType()
	{
		return m_msgtype;
	}
	
	
	public VSMsgID getMsgID()
	{
		return m_msgid;
	}
	
	//-- Returns null if no order was set
	public ViewId getViewID()
	{
		return m_viewid;
	}
	
	//-- Return -1 if no order was set
	public long getOrder()
	{
		return m_order;
	}
	
	
	@Override
	public int size()
	{
		return Global.BYTE_SIZE + m_msgid.size();
	}

	
	@Override
	public void readFrom(DataInputStream in)
					throws IOException, IllegalAccessException, InstantiationException

	{
		m_msgtype = VSTotalOrderMsgType.class.getEnumConstants()[ in.readByte() ];
		
		m_msgid = new VSMsgID();
		m_msgid.readFrom( in );
		isOrderingMessage = in.readBoolean();
		
		if( !in.readBoolean() )
		{
			m_viewid = null;
			m_order  = -1;
		}
		else
		{
			m_viewid = new ViewId();
			m_viewid.readFrom( in );
		
			m_order = in.readLong();
		}
	}

	
	@Override
	public void writeTo(DataOutputStream out) throws IOException
	{
		out.writeByte( m_msgtype.ordinal() );
		m_msgid.writeTo( out );
		out.writeBoolean(isOrderingMessage);
		
		if( m_viewid==null )
			out.writeBoolean( false );
		else
		{
			out.writeBoolean( true );
			m_viewid.writeTo( out );
			out.writeLong( m_order );
		}
	}
	
	
	private VSTotalOrderMsgType	m_msgtype;
	private VSMsgID				m_msgid;
	private ViewId				m_viewid;
	private long				m_order;
	
	// aufgabe 5.4
	private boolean				isOrderingMessage = false;
	
	public boolean isOrderingMessage() {
		return isOrderingMessage;
	}


	public void setOrderingMessage(boolean isOrderingMessage) {
		this.isOrderingMessage = isOrderingMessage;
	}


	private VSTotalOrderHeader(VSTotalOrderMsgType msgtype, VSMsgID msgid, ViewId viewid, long order)
	{
		if( msgid==null )
			throw new NullPointerException( "msgid" );
		
		m_msgtype = msgtype;
		m_msgid   = msgid;
		m_viewid  = viewid;
		m_order	  = order;
		
		// aufgabe 5.4
		isOrderingMessage = false;
	}
}
