%
% This is file `iacas.cls',
%
% It is based on aiaa package, modified for IACAS by Moshe Idan
%
%%
%%  The basic approach is to load the LaTeX article class and various
%%  standard LaTeX packages, and then make modifications and extensions to
%%  conform to IACAS's guidelines.
%%
%%  To use, simply select this class file in your LaTeX document.  For example,
%%
%%    \documentclass{iacas}
%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{iacas}[2012/01/14 IACAS technical conference paper]
\ProcessOptions
\LoadClass[a4paper,12pt]{article}
\RequirePackage[margin=1in,headheight=\baselineskip]{geometry}% page layout
\RequirePackage{graphicx}% images
\RequirePackage{array}%    extra row height
\RequirePackage{cite}% superscript, sorted, compressed citation commands
\setlength{\intextsep}{0.7\baselineskip \@plus 2\p@ \@minus 1\p@}
\def\fps@figure{htbp}
\def\fps@table{htbp}
\renewcommand\maketitle{\par
  \begingroup
    \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hb@xt@1.8em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
    \newpage
    \global\@topnum\z@   % Prevents figures from going at top of page.
    \@maketitle
    \@thanks
  \endgroup
  \setcounter{footnote}{0}%
  \renewcommand\thefootnote{\@alph\c@footnote}% change to alpha style footnote
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
\def\@maketitle{%
  \newpage
  \null
  \begin{center}%
    \vskip\baselineskip
    \vskip1em
  \let \footnote \thanks
    {\large\bfseries \@title \par}%
    \vskip 1.5em%
    {%\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
  \end{center}%
  \par
  \vskip 1.5ex}
\setlength\hfuzz{2pt}
\pagestyle{empty}
%\setkeys{Gin}{width=\linewidth,totalheight=\textheight,keepaspectratio}
\setlength{\extrarowheight}{2pt}
\renewenvironment{abstract}%
  {\small\quotation\bfseries}%
  {\normalfont\endquotation\normalsize}
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{\footnotesize\bfseries #1.~~#2}%
  \ifdim \wd\@tempboxa >\hsize
    \footnotesize\bfseries #1.~~#2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}
\setlength\belowcaptionskip{5pt plus 1pt minus 1pt}
\renewcommand\thesection      {\@Roman\c@section}
\renewcommand\thesubsection   {\@Alph\c@subsection}
\renewcommand\p@subsection{\thesection.}
\renewcommand\thesubsubsection{\@arabic\c@subsubsection}
\renewcommand\p@subsubsection{\thesection.\thesubsection.}
\renewcommand\theparagraph    {\@alph\c@paragraph}
\renewcommand\thesubparagraph {\@roman\c@subparagraph}
\setcounter{secnumdepth}{3}% no counters \paragraph and below
\def\@seccntformat#1{\csname the#1\endcsname.\quad}
\renewcommand\section{\@startsection {section}{1}{\z@}%
                                     {3.5ex \@plus -1ex \@minus -.2ex}%
                                     {2.3ex \@plus.2ex}%
                                     {\center\normalfont\normalsize\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                       {-3.25ex\@plus -1ex \@minus -.2ex}%
                                       {1.5ex \@plus .2ex}%
                                       {\normalfont\normalsize\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                       {-3.25ex\@plus -1ex \@minus -.2ex}%
                                       {1.5ex \@plus .2ex}%
                                       {\normalfont\normalsize\itshape}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                      {2.25ex \@plus1ex \@minus.2ex}%
                                      {-1em}%
                                      {\normalfont\normalsize\scshape}}
\renewcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                         {2.25ex \@plus1ex \@minus .2ex}%
                                         {-1em}%
                                         {\normalfont\normalfont}}
\renewcommand{\@biblabel}[1]{#1.\hspace{1ex}}
\renewenvironment{thebibliography}[1]%
     {\section*{\refname\@mkboth{\MakeUppercase\refname}%
                                 {\MakeUppercase\refname}}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\setlength{\leftmargin}{0pt}%
            \settowidth{\labelwidth}{\@biblabel{#1}}%
            \setlength{\itemindent}{\parindent}%
            \advance\itemindent by \labelwidth%
            \setlength{\labelsep}{0.0em}%
            \setlength{\itemsep}{-\smallskipamount}%
            \@openbib@code%
            \usecounter{enumiv}%
            \let\p@enumiv\@empty%
            \renewcommand\theenumiv{\@arabic\c@enumiv}%
            }
      \sloppy\clubpenalty4000\widowpenalty4000%
      \sfcode`\.\@m}
      {\def\@noitemerr
        {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
\AtBeginDocument{%
 \@ifundefined{TPT@hsize}{}{%
  \def\tablenotes{%
    \TPT@hsize
    \list{}%
         {\footnotesize\topsep4pt \partopsep\z@skip
          \itemsep2pt \parsep\z@ \itemindent\z@ \leftmargin1.5em
          \def\makelabel##1{\tnote{##1}\hfil}\labelwidth0.5em\labelsep1pt%
          \relax}}
 }%
 \@ifundefined{nomgroup}{}{%
  \RequirePackage{ifthen}% for typesetting groups within nomenclature
  \setlength{\nomitemsep}{-\parsep}
  \renewcommand{\nomgroup}[1]{%
   \medskip%
   \ifthenelse{\equal{#1}{A}}%
    {\relax}%
    {\ifthenelse{\equal{#1}{G}}%
      {\item []\hspace*{-\leftmargin}%
       \emph{Symbols}\hfill\smallskip}%
      {\ifthenelse{\equal{#1}{B}}%
        {\item []\hspace*{-\leftmargin}%
         \emph{Subscripts}\hfill\smallskip}%
        {\ifthenelse{\equal{#1}{P}}%
          {\item []\hspace*{-\leftmargin}%
           \emph{Superscripts}\hfill\smallskip}%
          {\ifthenelse{\equal{#1}{R}}%
            {\item []\hspace*{-\leftmargin}%
             \emph{Abbreviations}\hfill\smallskip}{}%
          }%
        }%
      }%
    }%
  }% \nomgroup
 }% @ifundefined
}% \AtBeginDocument
\endinput
%%
%% End of file `iacas.cls'.
