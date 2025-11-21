# Makefile per compilazione LaTeX del report
# Usage:
#   make          - compila il PDF
#   make clean    - rimuove file accessori
#   make all      - compila e pulisce
#   make view     - apre il PDF dopo la compilazione

# Nome del file principale (senza estensione)
TARGET = First_Assignment_Nicolas_Leone_1986354

# Compilatore LaTeX (pdflatex o xelatex)
LATEX = pdflatex

# Opzioni compilatore
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

# File da generare
PDF = $(TARGET).pdf

# File accessori da eliminare
AUX_FILES = *.aux *.log *.out *.toc *.bbl *.blg *.lof *.lot *.fls *.fdb_latexmk \
            *.synctex.gz *.nav *.snm *.vrb *.bcf *.run.xml *.idx *.ilg *.ind

# Target di default: compila e pulisce
all: $(PDF) clean

# Compila il PDF (doppia passata per risolvere riferimenti)
$(PDF): $(TARGET).tex
	@echo "📄 Compilazione in corso..."
	@$(LATEX) $(LATEX_FLAGS) $(TARGET).tex > /dev/null
	@$(LATEX) $(LATEX_FLAGS) $(TARGET).tex > /dev/null
	@echo "✅ PDF generato: $(PDF)"

# Compila con bibliografia (se presente .bib)
bib: $(TARGET).tex
	@echo "📚 Compilazione con bibliografia..."
	@$(LATEX) $(LATEX_FLAGS) $(TARGET).tex > /dev/null
	@bibtex $(TARGET) > /dev/null 2>&1
	@$(LATEX) $(LATEX_FLAGS) $(TARGET).tex > /dev/null
	@$(LATEX) $(LATEX_FLAGS) $(TARGET).tex > /dev/null
	@echo "✅ PDF con bibliografia generato: $(PDF)"

# Rimuove tutti i file accessori
clean:
	@rm -f $(AUX_FILES)
	@echo "🧹 File accessori rimossi"

# Rimuove anche il PDF
distclean: clean
	@echo "Rimozione PDF..."
	@rm -f $(PDF)
	@echo "✅ Tutto pulito"

# Apre il PDF con il visualizzatore di default (macOS)
view: $(PDF)
	@echo "Apertura PDF..."
	@open $(PDF)

# Compila in modalità watch (ricompila ad ogni modifica)
watch:
	@echo "Modalità watch attiva (Ctrl+C per fermare)..."
	@while true; do \
		make -s $(PDF); \
		sleep 2; \
	done

# Help
help:
	@echo "Comandi disponibili:"
	@echo "  make          - Compila il PDF e pulisce file accessori"
	@echo "  make all      - Come make"
	@echo "  make bib      - Compila con bibliografia (BibTeX)"
	@echo "  make clean    - Rimuove solo file accessori"
	@echo "  make distclean- Rimuove file accessori e PDF"
	@echo "  make view     - Apre il PDF dopo compilazione"
	@echo "  make watch    - Ricompila automaticamente ad ogni modifica"
	@echo "  make help     - Mostra questo messaggio"

# Phony targets (non sono file reali)
.PHONY: all clean distclean view watch help bib
