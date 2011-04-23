%%
%% This is file `lmedoc.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% lmedoc.dtx  (with options: `package')
%% 
%% Copyright (C) 2000 by Dietrich Paulus. All rights reserved.
%% 
%% 
%% This file is part of the LME TeX package
%% 
%% IMPORTANT NOTICE:
%% 
%% You are not allowed to change this file.  You may however copy this file
%% to a file with a different name and then change the copy.
%% 
%% You are NOT ALLOWED to distribute this file alone.  You are NOT ALLOWED
%% to take money for the distribution or use of this file (or a changed
%% version) except for a nominal charge for copying etc.
%% 
%% You are allowed to distribute this file under the condition that it is
%% distributed together with all files mentioned below.
%% 
%% If you receive only some of these files from someone, complain!
%% 
%% Error Reports in case of UNCHANGED versions to
%% 
%%                         Dietrich Paulus
%%                         Lehrstuhl f. Mustererkennung
%%                         D 91054 Erlangen
%%             Internet:  <info@immd4.cs.fau.de>
%% 
%% $Id: lmedoc.dtx,v 1.13 2003/02/13 09:29:00 zobel Exp $
%% Copyright (C) 2000 Dietrich Paulus
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{lmedoc}[2002/13/02 v1.0 lme document (LME)]
\def\BaseClass{report}
\LoadClass[twoside,12pt]{\BaseClass}     % 12 pt base class report
\RequirePackage{a4wide}          % page layout
\RequirePackage{rcs}             % revision control is helpful
\RequirePackage{bm}        % standard math notation (fonts)
\RequirePackage{fixmath}        % standard math notation (fonts)
\RequirePackage{amsmath}            % standard math notation (vectors/sets/...)
\RequirePackage{math}            % standard math notation (vectors/sets/...)
\RequirePackage{graphicx}        % eps graphics support
\RequirePackage{times}           % scalable fonts
\RequirePackage{setspace}  % set correct baselinestretch
\RequirePackage{twolang}         % will load german.sty
\def\Type{0}
\DeclareOption{german}{\selectlanguage{\german}}
\DeclareOption{english}{\selectlanguage{\english}}
\DeclareOption{da}{\gdef\Type{1}}
\DeclareOption{sa}{\gdef\Type{2}}
\DeclareOption{mt}{\gdef\Type{3}}
\DeclareOption{bt}{\gdef\Type{4}}
\DeclareOption{rep}{\gdef\Type{5}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\BaseClass}}
\ProcessOptions\relax
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\typeout{Type \Type}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\onehalfspacing
\let\oldChapter=\chapter
\def\chapter{\cleardoublepage\oldChapter}
\pagestyle{headings}
\DeclareGraphicsRule{.ps.gz}{eps}{.ps.bb}{`gunzip -c #1}
\DeclareGraphicsRule{.cps.gz}{eps}{.cps.bb}{`gunzip -c #1}
\DeclareGraphicsRule{.eps.gz}{eps}{.eps.bb}{`gunzip -c #1}
\DeclareGraphicsRule{.ps.Z}{eps}{.ps.bb}{`gunzip -c #1}
\DeclareGraphicsRule{.cps.Z}{eps}{.cps.bb}{`gunzip -c #1}
\DeclareGraphicsRule{.eps.Z}{eps}{.eps.bb}{`gunzip -c #1}
\DeclareGraphicsExtensions{.ps,.eps,.ps.Z,.eps.Z,.ps.gz,.eps.gz,.ps.bb,.eps.bb}
\makeatletter
\def\city#1{\def\@city{#1}}
\def\birthdate#1{\def\@birthdate{#1}}
\def\advisor#1{\def\@advisor{#1}}
\def\startthesis#1{\def\@startthesis{#1}}
\def\endthesis#1{\def\@endthesis{#1}}
\def\deckblatt{\bgroup\def\baselinestretch{1.0}%
    \def\Titel##1{\gdef\@Titel{##1}\typeout{Defining Titel}}
    \def\Name##1{\gdef\@Name{##1}}
    \def\Vorname##1{\gdef\@Vorname{##1}}
    \def\Geburtsort##1{\gdef\@Geburtsort{##1}}
    \def\Geburtsdatum##1{\gdef\@Geburtsdatum{##1}}
    \def\Betreuer##1{\gdef\@Betreuer{##1}}
    \def\Start##1{\gdef\@Start{##1}}
    \def\Ende##1{\gdef\@Ende{##1}}
}
\def\enddeckblatt{%
\clearpage
\thispagestyle{empty}
\vspace*{1.6cm}
\begin{center}
\Large
{\bf \@Titel}\\[20mm]
\large
{\bf
    \ifnum\Type=1 Diplomarbeit\fi
    \ifnum\Type=2 Studienarbeit\fi
    \ifnum\Type=3 Masters Thesis\fi
    \ifnum\Type=4 Bachelor Thesis\fi
    \ifnum\language=\l@english
 \ in Computer Science
    \else
 \ im Fach Informatik\fi
    } \\[10mm]
\normalsize
\ifnum\language=\l@english submitted \else vorgelegt \fi \\
\ifnum\language=\l@english by \else von \fi \\
\vskip 0.8cm plus 0.2cm minus 0.5cm\relax
\end{center}
\begin{center}
\@Vorname\ \@Name \\[2mm]
\ifnum\language=\l@english born \else Geboren am\fi~
\@Geburtsdatum\ in \@Geburtsort \\% [45mm]
\vfill
\end{center}
\begin{center}
\ifnum\language=\l@english\relax
  Written at
\else
  Angefertigt am
\fi
\\[5mm]
Lehrstuhl f"ur Mustererkennung (Informatik 5)\\
Institut f"ur Informatik \\
Friedrich-Alexander-Universit"at Erlangen-N"urnberg.\\[5mm]
\@ifundefined{ZweitInstitut}{}{%
\ifnum\language=\l@english\relax
  in Cooperation with
\else
  in Zusammenarbeit mit
\fi
\\[5mm]
  \ZweitInstitut\\[10mm]
}
\end{center}
\begin{flushleft}
\ifnum\language=\l@english\relax Advisor\else Betreuer\fi: \@Betreuer \\[2mm]
\ifnum\language=\l@english\relax Started\else Beginn der Arbeit\fi: \@Start \\[2mm]
\ifnum\language=\l@english\relax Finished\else Abgabe der Arbeit\fi: \@Ende \\
\end{flushleft}
\clearpage
\egroup}
\endinput
%%
%% End of file `lmedoc.cls'.
