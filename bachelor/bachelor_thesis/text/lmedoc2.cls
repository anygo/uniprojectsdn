% lmedoc2.cls
% Martin Gropp (2010-02-09)
% Based on lmedoc.cls by Dietrich Paulus

% Significant changes:
% * KomaScript base class
% * Babel support
% * XeTeX support
% * Option error handling
% * sans-serif title option
% * ...
%
% ----------------------------------------------------------------------
% In order to maintain compatibility with XeTeX, utf8 encoding
% is mandatory, in non-XeTeX inputenc/utf8 is loaded automatically.
% ----------------------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{lmedoc2}[2010/02/03 v0.1 lme document 2 (LME)]

% Base class: scrreprt with 12pt,a4paper.
% twoside,openright should be specified in the document, not here.
\def\BaseClass{scrreprt}
\LoadClass[12pt,a4paper]{\BaseClass}

% Packages
\RequirePackage{titlesec}
\RequirePackage{a4wide}          % page layout
\RequirePackage{rcs}             % revision control is helpful
\RequirePackage{amsmath}         % standard math notation (vectors/sets/...)
\RequirePackage{amstext}
\RequirePackage{amssymb}
\RequirePackage{math}            % standard math notation (vectors/sets/...)
\RequirePackage{graphicx}        % graphics support
\RequirePackage{setspace}        % set correct baselinestretch

% XeTeX uses UTF8 encoding by default
% (but does not necessarily load a proper font, so we load fontspec),
% other LaTeXes probably need inputenc.
\RequirePackage{ifxetex}         % Detect XeTeX
\ifxetex\usepackage{fontspec}\else\usepackage[utf8]{inputenc}\fi

% Options
\def\lme@type{0}
\DeclareOption{german}{\gdef\lme@language{0}}
%\DeclareOption{ngerman}{\gdef\lme@language{0}}
\DeclareOption{english}{\gdef\lme@language{1}}
\DeclareOption{da}{\gdef\lme@type{1}}
\DeclareOption{sa}{\gdef\lme@type{2}}
\DeclareOption{mt}{\gdef\lme@type{3}}
\DeclareOption{bt}{\gdef\lme@type{4}}
\DeclareOption{rep}{\gdef\lme@type{5}}
\DeclareOption{sftitle}{\gdef\lme@sftitle}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\BaseClass}}
\ProcessOptions\relax

% Load babel
\RequirePackage[ngerman,english]{babel}
\RequirePackage[nodayofweek]{datetime}

% Check thesis type
\ifnum\lme@type=0\ClassError{lmedoc2}{No thesis type specified}{Add one of da,sa,mt,bt,rep to the document class options.}\fi

% Check language
\ifdefined\lme@language\else\ClassError{lmedoc2}{Missing language option.}{Add either english or german to the document class options.}\fi

% Set main language
% This is necessary because we load babel with two languages, and
% the first language (= the one not used) will be the one specified
% by the user in our options...
\ifnum\lme@language=0\main@language{ngerman}\else\main@language{english}\fi

% Schusterjungen & Hurenkinder
\clubpenalty=10000
\widowpenalty=10000
\displaywidowpenalty=10000

% Line spacing
\onehalfspacing

% Numbering
\setcounter{secnumdepth}{2}
\renewcommand{\thepart}{\Roman{part}.}
\renewcommand{\thechapter}{\arabic{chapter}.}
\renewcommand{\thesection}{\arabic{chapter}.\arabic{section}}
\renewcommand{\thesubsection}{\arabic{chapter}.\arabic{section}.\arabic{subsection}}
\renewcommand{\thefigure}{\thechapter\arabic{figure}}
\renewcommand{\thetable}{\thechapter\arabic{table}}
\renewcommand{\theequation}{\thechapter\arabic{equation}}

% Add \cleardoublepage to \chapter
\let\oldchapter=\chapter
\def\chapter{\cleardoublepage\oldchapter}

% Add \filbreak to structure commands
\let\oldsection=\section
\def\section{\ifnum\value{section}>0 \filbreak\fi\oldsection}
\let\oldsubsection=\subsection
\def\subsection{\ifnum\value{subsection}>0 \filbreak\fi\oldsubsection}

% Deckblatt
\pagestyle{headings}
\def\city#1{\def\@city{#1}}
\def\birthdate#1{\def\@birthdate{#1}}
\def\advisor#1{\def\@advisor{#1}}
\def\startthesis#1{\def\@startthesis{#1}}
\def\endthesis#1{\def\@endthesis{#1}}

\newenvironment{deckblatt}{ %
  \begingroup
  \def\baselinestretch{1.0}
  \def\Titel##1{\gdef\@Titel{##1}}
  \def\Name##1{\gdef\@Name{##1}}
  \def\Vorname##1{\gdef\@Vorname{##1}}
  \def\Geburtsort##1{\gdef\@Geburtsort{##1}}
  \def\Geburtsdatum##1{\gdef\@Geburtsdatum{##1}}
  \def\Betreuer##1{\gdef\@Betreuer{##1}}
  \def\Start##1{\gdef\@Start{##1}}
  \def\Ende##1{\gdef\@Ende{##1}}
}{ %
  \ifdefined\lme@sftitle\sf\fi
  \clearpage
  \thispagestyle{empty}
  \vspace*{1.6cm}
  \begin{center}
  \ifnum\lme@language=0\otherlanguage{ngerman}\else\otherlanguage{english}\fi
  \Large\textbf{\@Titel} \\[20mm]
  \large\textbf{
    \ifnum\lme@type=1 Diplomarbeit\fi
    \ifnum\lme@type=2 Studienarbeit\fi
    \ifnum\lme@type=3 Masters Thesis\fi
    \ifnum\lme@type=4 Bachelor Thesis\fi
    \iflanguage{english}{ in Computer Science}{ im Fach Informatik}
  } \\[10mm] %
  %
  \normalsize
  \iflanguage{english}{submitted}{vorgelegt} \\
  \iflanguage{english}{by}{von} \\
  \vskip 0.8cm plus 0.2cm minus 0.5cm\relax
  \@Vorname\ \@Name \\[2mm]
  \iflanguage{english}{born}{geboren am}\ \@Geburtsdatum\ in \@Geburtsort \\
  \vfill %
  %
  \iflanguage{english}{Written at}{Angefertigt am}
  \\[5mm]
  Lehrstuhl für Mustererkennung (Informatik 5)\\
  Department Informatik \\
  Friedrich-Alexander-Universität Erlangen-Nürnberg.\\[5mm]
  \@ifundefined{ZweitInstitut}{}{%
  \iflanguage{english}{in cooperation with}{in Zusammenarbeit mit}
  \\[5mm]
    \ZweitInstitut\\[10mm]
  }
  \end{center} %
  %
  \begin{flushleft}
  \vspace{5ex}
  \iflanguage{english}{Advisor}{Betreuer}: \@Betreuer \\[2mm]
  \iflanguage{english}{Started}{Beginn der Arbeit}: \@Start \\
  \iflanguage{english}{Finished}{Abgabe der Arbeit}: \@Ende \\
  \end{flushleft}
  \clearpage
  \endotherlanguage
  \endgroup
}

% Erklärung
\newcommand{\thesisstatement}{\begin{otherlanguage}{ngerman}
Ich versichere, dass ich die Arbeit ohne fremde Hilfe und ohne Benutzung
anderer als der angegebenen Quellen angefertigt habe und dass die Arbeit
in gleicher oder ähnlicher Form noch keiner anderen Prüfungsbehörde
vorgelegen hat und von dieser als Teil einer Prüfungsleistung
angenommen wurde. Alle Ausführungen, die wörtlich oder sinngemäß
übernommen wurden, sind als solche gekennzeichnet.

Die Richtlinien des Lehrstuhls für Studien- und Diplomarbeiten
habe ich gelesen und anerkannt, insbesondere die Regelung des
Nutzungsrechts. \\[15mm]
Erlangen, den \today
\end{otherlanguage}}

\endinput
