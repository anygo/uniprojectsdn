package vsue.totalorder;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.concurrent.Semaphore;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.logging.Logger;

import org.jgroups.Address;
import org.jgroups.ChannelException;
import org.jgroups.JChannel;
import org.jgroups.Message;
import org.jgroups.ReceiverAdapter;
import org.jgroups.View;
import org.jgroups.conf.ClassConfigurator;

public class VSTestClient
{
	public static void main(String[] args) throws Exception
	{
		int		id    = 0;
		int		mbrs  = 0;
		String	group = null;
		
		try
		{
			id    = Integer.parseInt( args[ 0 ] );
			mbrs  = Integer.parseInt( args[ 1 ] );
			group = args[ 2 ];
			
			if( group.isEmpty() )
				throw new IllegalArgumentException();
		}
		catch( Exception e )
		{
			System.err.println( "Usage: TestClient <id> <host_count> <groupname>" );
			System.exit( 1 );
		}
		
		//-- IPv6 is sometimes broken with JGroups, disable it...
		System.setProperty( "java.net.preferIPv4Stack", "true" );
		ClassConfigurator.addProtocol( (short) 2012, VSTotalOrder.class );

		VSTestClient client = new VSTestClient( id );
		
		m_logger.info( "Connect to group " + group );
		client.connect( group, mbrs );
		
		m_logger.info( "Spam ..." );
		client.spamChannel( 100 );

		m_logger.info( "Wait for messages" );
		client.waitForMessages( 100 * mbrs );
		
		m_logger.info( "Wait a little bit further ..." );
		Thread.sleep( 5000 );

		m_logger.info( "Disconnect" );
		client.disconnect();
	}
	
	
	public VSTestClient(int id)
	{
		m_id		= id;
		m_viewlock 	= new ReentrantLock();
		m_viewsig 	= m_viewlock.newCondition();
	}
	
	
	public void connect(String cluster_name, int minmbrs) throws IOException, ChannelException
	{
		if( m_channel!=null )
			throw new IllegalStateException( "Client is already connected." );

		if( cluster_name==null || cluster_name.isEmpty() )
			throw new IllegalArgumentException( "cluster_name must not be null or empty." );
		if( minmbrs<0 )
			throw new IllegalArgumentException( "minmbrs must not be negative." );
		
		//-- The log has to be started first, since it is used by the message handler
		m_logrecv = new PrintWriter( new FileWriter( "recv.log." + m_id ) );
		
		m_channel = new JChannel( "stack.xml" );
		m_channel.setReceiver( new MessageReceiver() );

		m_logger.info( "Connect to cluster " +  cluster_name );
		m_channel.connect( cluster_name );
		
		m_viewlock.lock();
		
		try
		{
			m_logger.info( "Wait for all cluster members" );
			
			while( m_view==null || m_view.getMembers().size()<minmbrs )
			{
				m_viewsig.awaitUninterruptibly();

				m_logger.info( "New View:" );
				for( Address addr : m_view.getMembers() )
					m_logger.info( "  " + addr.toString() );
			}
		}
		finally
		{
			m_viewlock.unlock();
		}
	}
	
	
	public void spamChannel(int count) throws ChannelException, InterruptedException
	{
		if( m_channel==null )
			throw new IllegalStateException( "Client is not connected." );
		
		for( int i=0; i<count; i++ )
		{
			Thread.sleep( (long) (Math.random() * 5) );
						
			m_channel.send( null, null, i );
		}
	}
	
	
	public void waitForMessages(int count)
	{
		m_semrecv.acquireUninterruptibly( count );
	}
	
	
	public void disconnect()
	{
		if( m_channel!=null )
		{
			m_channel.close();
			m_channel = null;
		}
		
		if( m_logrecv!=null )
		{
			m_logrecv.close();
			m_logrecv = null;
		}
	}
	
	
	private final static Logger m_logger = Logger.getLogger( VSTestClient.class.getName() );
	
	private ReentrantLock	m_viewlock;
	private Condition		m_viewsig;
	private Semaphore		m_semrecv = new Semaphore( 0 );
	
	private int				m_id;
	
	private PrintWriter		m_logrecv;
	private JChannel		m_channel;
	private View			m_view;
	
	
	private class MessageReceiver extends ReceiverAdapter
	{
		@Override
		public void receive(Message msg)
		{
			m_semrecv.release();
			
			String logmsg = "Received message from " + msg.getSrc() + ": " + msg.getObject();
			m_logger.info( logmsg );
			m_logrecv.println( logmsg );
		}

		
		@Override
		public void viewAccepted(View view)
		{
			m_viewlock.lock();
			
			try
			{
				m_view = view;
				
				m_viewsig.signalAll();
			}
			finally
			{
				m_viewlock.unlock();
			}
		}
	}
}