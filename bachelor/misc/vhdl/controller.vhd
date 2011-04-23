-- vim: set syntax=vhdl ts=8 sts=2 sw=2 et:

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.all;
USE IEEE.STD_LOGIC_ARITH.all;
USE IEEE.STD_LOGIC_UNSIGNED.all;

ENTITY controller IS PORT (
  S                                      : IN  std_logic_vector (1 TO 5);
  DISPLAY0, DISPLAY1, DISPLAY2, DISPLAY3 : OUT integer RANGE 0 TO 9;
  alarm                                  : OUT std_logic;
  clk                                    : IN  std_logic;
  reset                                  : IN  std_logic );
END controller;

ARCHITECTURE Behavioral OF controller IS

  COMPONENT Display60 IS PORT (
    number               : IN  integer RANGE 0 TO 59;
    position1, position0 : OUT integer RANGE 0 TO 9);
  END COMPONENT;

  COMPONENT Display24 IS PORT (
    number               : IN  integer RANGE 0 TO 23;
    position1, position0 : OUT integer RANGE 0 TO 9);
  END COMPONENT;


  SIGNAL timer      : std_logic_vector (13 DOWNTO 0);
  SIGNAL sectrigger : std_logic; -- '1' fuer einen Takt wenn eine Sekunde
                                 -- vergangen ist, sonst '0'
  SIGNAL DISPLAY10, DISPLAY11, DISPLAY20, DISPLAY21 : integer RANGE 0 TO 9;
  SIGNAL DISPLAY30, DISPLAY31, DISPLAY40, DISPLAY41 : integer RANGE 0 TO 9;

  SIGNAL hours, whours     : integer RANGE 0 TO 23;
  SIGNAL secs, mins, wmins : integer RANGE 0 TO 59;

  type state_type IS (ntime, set_time, set_alarm);

  SIGNAL current_state : state_type;

BEGIN
  SECTIMER : PROCESS(clk, reset)
    VARIABLE timer_msb_to_zero : std_logic_vector(timer'RANGE);
  BEGIN
    IF reset = '1' THEN
      timer <= (others => '0');
    ELSIF clk'event and clk = '1' THEN
      timer_msb_to_zero                 := timer;
      -- set msb TO zero, the msb will be our overflow bit
      timer_msb_to_zero(timer'length-1) := '0';
      timer <= timer_msb_to_zero + 1;
    END IF;
  END PROCESS SECTIMER;

  sectrigger <= timer(timer'length-1);

  FSM : PROCESS (clk, reset, current_state, secs, mins, hours, wmins, whours, S)
    VARIABLE next_state : state_type;
  BEGIN
    IF reset = '1' THEN
      hours         <= 0; mins <= 0; secs <= 0;
      whours        <= 0; wmins <= 0; alarm <= '0';
      current_state <= ntime;
    ELSIF (clk'event and clk = '1') THEN

      IF (not (current_state = set_time) and (sectrigger = '1')) THEN
        -- Zaehle Uhr hoch
        IF secs = 59 THEN
          secs      <= 0;
          IF mins = 59 THEN
            mins    <= 0;
            IF hours = 23 THEN
              hours <= 0;
            ELSE
              hours <= hours+1;
            END IF;
          ELSE
            mins    <= mins+1;
          END IF;
        ELSE
          secs      <= secs+1;
        END IF;
      END IF;

      CASE current_state IS
        -- Zustand Time
        WHEN ntime     =>
          IF (whours = hours and wmins = mins and S(5) = '1') THEN
            alarm <= '1';
          ELSE
            alarm <= '0';
          END IF;

        -- Zustand SetTime
        WHEN set_time  =>
          IF (sectrigger = '1' and S(4) = '1') THEN
            IF (hours = 23) THEN
              hours <= 0;
            ELSE
              hours <= hours + 1;
            END IF;
          END IF;

          IF (sectrigger = '1' and S(3) = '1') THEN
            IF (mins = 59) THEN
              mins <= 0;
            ELSE
              mins <= mins + 1;
            END IF;
          END IF;

        -- Zustand SetAlarm
        WHEN set_alarm =>
          IF (sectrigger = '1' and S(4) = '1') THEN
            IF (whours = 23) THEN
              whours <= 0;
            ELSE
              whours <= whours + 1;
            END IF;
          END IF;

          IF (sectrigger = '1' and S(3) = '1') THEN
            IF (wmins = 59) THEN
              wmins <= 0;
            ELSE
              wmins <= wmins + 1;
            END IF;
          END IF;

        -- Illegale Zustaende
        WHEN others    =>
          next_state <= ntime;

      END CASE;

      -- Setze naechsten Zustand
      IF ('1' = S(1) and ('1' = S(2))) THEN
        next_state := ntime;
      ELSIF '1' = S(1) THEN
        next_state := set_time;
      ELSIF '1' = S(2) THEN
        next_state := set_alarm;
      ELSE
        next_state := ntime;
      END IF;
      -- Setze naechsten Zustand
      current_state <= next_state;
    END IF;

  END PROCESS FSM;

  -- MINUTENANZEIGE (ZEIT)
  MIN_CONVERT : Display60
    PORT MAP (number => mins, position1 => DISPLAY11, position0 => DISPLAY10);

  -- STUNDENANZEIGE (ZEIT)
  HR_CONVERT : Display24
    PORT MAP (number => hours, position1 => DISPLAY21, position0 => DISPLAY20);

  -- MINUTENANZEIGE (WECKZEIT)
  WMIN_CONVERT : Display60
    PORT MAP (number => wmins, position1 => DISPLAY31, position0 => DISPLAY30);

  -- STUNDENANZEIGE (WECKZEIT)
  WHR_CONVERT : Display24
    PORT MAP (number => whours, position1 => DISPLAY41, position0 => DISPLAY40);

  -- Beschreibung, wie die DISPLAY-Variablen gesetzt werden
  Switch_Display : PROCESS (
    current_state,
    DISPLAY10, DISPLAY11, DISPLAY20, DISPLAY21, DISPLAY30,
    DISPLAY31, DISPLAY40, DISPLAY41 ) IS
  BEGIN
    IF current_state /= set_alarm THEN
      DISPLAY0 <= DISPLAY10;
      DISPLAY1 <= DISPLAY11;
      DISPLAY2 <= DISPLAY20;
      DISPLAY3 <= DISPLAY21;
    ELSE
      DISPLAY0 <= DISPLAY30;
      DISPLAY1 <= DISPLAY31;
      DISPLAY2 <= DISPLAY40;
      DISPLAY3 <= DISPLAY41;
    END IF;
  END PROCESS Switch_Display;

END Behavioral;
