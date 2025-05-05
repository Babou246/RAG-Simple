import csv
import datetime
import json
import random
import tkinter as tk
from tkinter import filedialog
import tkinter.font as tkfont
import threading
import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import time
import customtkinter as ctk
import traceback
import re

# Imports LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Chargement des variables d'environnement
load_dotenv()

# Configuration pour l'API
API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_BASE_URL = os.environ.get("OPENROUTER_API_BASE_URL", "XXXXXXX")
MODEL_NAME = os.environ.get("OPENROUTER_MODEL_NAME", "XXXXXX")

# Vérification simple des clés API
if not API_KEY:
    print("Attention: La variable d'environnement OPENROUTER_API_KEY n'est pas définie.")
    print("Certaines fonctionnalités IA ne seront pas disponibles.")


class ModernRAGApp:
    def __init__(self):
        # Configuration de customtkinter
        ctk.set_appearance_mode("Light")
        ctk.set_default_color_theme("blue")

        # Variables de l'application
        self.uploaded_files = []
        self.session_id = str(uuid.uuid4())
        self.conversation_chain = None
        self.processing = False
        self.history = []

        # Variables pour la barre latérale repliable
        self.sidebar_expanded = True
        self.sidebar_width_expanded = 250
        self.sidebar_width_collapsed = 50

        # Variables pour les paramètres de traitement et modèle
        self.chunk_size = 1500
        self.chunk_overlap = 150
        self.search_k = 4
        self.temperature = 0.7

        # Créer les dossiers si nécessaire
        self.upload_folder = Path("uploads")
        self.upload_folder.mkdir(exist_ok=True)

        self.history_folder = Path("history")
        self.history_folder.mkdir(exist_ok=True)

        # Variable pour stocker la fenêtre des paramètres
        self.settings_window = None

        # Création de la fenêtre principale
        self.app = ctk.CTk()
        self.app.title("DocChat AI")
        self.app.geometry("1000x700")
        self.app.minsize(900, 600)

        # Création de l'interface
        self.setup_ui()

        # Charger l'historique au démarrage
        self.load_history()

    def setup_ui(self):
        # Utilisation de grid pour la structure principale

        # Frame principale - Définir la couleur de fond principale
        self.main_container = ctk.CTkFrame(self.app, fg_color="#b4d1e8", corner_radius=0)
        self.main_container.pack(fill="both", expand=True)

        # Barre supérieure (Header) - Utiliser une couleur pour le header
        self.header_frame = ctk.CTkFrame(self.main_container, corner_radius=0, fg_color="#f0f3f6", height=50)
        self.header_frame.pack(fill="x", side="top")

        # Logo et titre - Ajuster la couleur du texte si nécessaire pour la lisibilité
        self.title_label = ctk.CTkLabel(self.header_frame, text="DocChat AI", font=ctk.CTkFont(size=20, weight="bold"), text_color="#333333")
        self.title_label.pack(side="left", padx=20)

        # Container pour sidebar et zone de chat - Transparent pour laisser la couleur du main_container
        self.content_container = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.content_container.pack(fill="both", expand=True, padx=0, pady=0)

        # Configurer le grid
        self.content_container.grid_columnconfigure(1, weight=1)
        self.content_container.grid_rowconfigure(0, weight=1)

        # Sidebar (panneau gauche) - Utiliser une couleur pour la sidebar
        self.sidebar = ctk.CTkFrame(self.content_container, width=self.sidebar_width_expanded, fg_color="#f0f3f6", corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nswe", padx=0, pady=0)

        # Zone de conversation (droite) - Le chat_area aura sa propre couleur de fond
        self.conversation_frame = ctk.CTkFrame(self.content_container, fg_color="transparent")
        self.conversation_frame.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)

        # Barre de statut (bas) - Peut conserver une couleur différente ou être harmonisée
        self.status_frame = ctk.CTkFrame(self.main_container, corner_radius=0, fg_color="#e0e0e0", height=25)
        self.status_frame.pack(fill="x", side="bottom")

        # Configurer le contenu
        self.setup_sidebar()
        self.setup_chat_area()
        self.setup_status_bar()

    def setup_sidebar(self):
        for widget in self.sidebar.winfo_children():
            widget.destroy()

        top_buttons_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        top_buttons_frame.pack(fill="x", pady=(10, 20), padx=5)

        self.menu_toggle_btn = ctk.CTkButton(
            top_buttons_frame,
            text="☰",
            command=self.toggle_sidebar,
            width=40, height=40,
            fg_color="transparent",
            hover_color="#d0d3d6",
            text_color="#333333"
        )
        self.menu_toggle_btn.pack(side="left", padx=5)

        self.new_chat_btn = ctk.CTkButton(
            top_buttons_frame,
            text=" + Nouveau Chat" if self.sidebar_expanded else "+",
            command=self.reset_conversation,
            width=self.sidebar_width_expanded - 60 if self.sidebar_expanded else 40,
            height=40,
            corner_radius=20 if self.sidebar_expanded else 0,
            fg_color="#f0f3f6",
            hover_color="#d0d3d6",
            text_color="#333333"
        )
        self.new_chat_btn.pack(side="left", expand=True, fill="x", padx=5)

        bottom_buttons_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        bottom_buttons_frame.pack(fill="x", side="bottom", pady=10, padx=5)

        self.upload_btn = ctk.CTkButton(
            bottom_buttons_frame,
            text=" ⬆️ Uploader Fichiers" if self.sidebar_expanded else "⬆️",
            command=self.select_files,
             width=40, height=40,
            fg_color="#f0f3f6",
            hover_color="#d0d3d6",
            text_color="#333333"
        )
        self.upload_btn.pack(side="top", pady=5, fill="x")

        self.process_btn = ctk.CTkButton(
            bottom_buttons_frame,
            text=" ✨ Traiter Documents" if self.sidebar_expanded else "✨",
            command=self.process_files,
             width=40, height=40,
            fg_color="#f0f3f6",
            hover_color="#d0d3d6",
            text_color="#333333"
        )
        self.process_btn.pack(side="top", pady=5, fill="x")

        self.reset_btn = ctk.CTkButton(
            bottom_buttons_frame,
            text=" 🔄 Tout Réinitialiser" if self.sidebar_expanded else "🔄",
            command=self.reset_conversation,
             width=40, height=40,
            fg_color="#f0f3f6",
            hover_color="#d0d3d6",
            text_color="#333333"
        )
        self.reset_btn.pack(side="top", pady=5, fill="x")

        self.history_btn = ctk.CTkButton(
            bottom_buttons_frame,
            text=" 📄 Historique" if self.sidebar_expanded else "📄",
            command=self.show_history,
             width=40, height=40,
            fg_color="transparent",
            hover_color="#d0d3d6",
            text_color="#333333"
        )
        self.history_btn.pack(side="top", pady=5, fill="x")

        self.help_btn = ctk.CTkButton(
            bottom_buttons_frame,
            text=" ❓ Aide" if self.sidebar_expanded else "❓",
            command=self.show_help,
             width=40, height=40,
            fg_color="transparent",
            hover_color="#d0d3d6",
            text_color="#333333"
        )
        self.help_btn.pack(side="top", pady=5, fill="x")

        self.settings_btn = ctk.CTkButton(
            bottom_buttons_frame,
            text=" ⚙️ Paramètres" if self.sidebar_expanded else "⚙️",
            command=self.show_settings,
             width=40, height=40,
            fg_color="transparent",
            hover_color="#d0d3d6",
            text_color="#333333"
        )
        self.settings_btn.pack(side="top", pady=5, fill="x")


    def toggle_sidebar(self):
        """Replie ou déplie la barre latérale"""
        if self.sidebar_expanded:
            # Replier
            self.sidebar.configure(width=self.sidebar_width_collapsed)
            self.content_container.grid_columnconfigure(0, minsize=self.sidebar_width_collapsed)

            # Cacher le texte des boutons et ajuster la largeur
            self.menu_toggle_btn.configure(text="☰", width=40)
            self.new_chat_btn.configure(text="+", width=40, corner_radius=0)
            self.upload_btn.configure(text="⬆️", width=40)
            self.process_btn.configure(text="✨", width=40)
            self.reset_btn.configure(text="🔄", width=40)
            self.history_btn.configure(text="📄", width=40)
            self.help_btn.configure(text="❓", width=40)
            self.settings_btn.configure(text="⚙️", width=40)

        else:
            # Déplier
            self.sidebar.configure(width=self.sidebar_width_expanded)
            self.content_container.grid_columnconfigure(0, minsize=self.sidebar_width_expanded)

            # Afficher le texte des boutons et ajuster la largeur
            self.menu_toggle_btn.configure(text="☰", width=40)
            self.new_chat_btn.configure(text=" + Nouveau Chat", width=self.sidebar_width_expanded - 60, corner_radius=20)
            self.upload_btn.configure(text=" ⬆️ Uploader Fichiers", width=self.sidebar_width_expanded - 10)
            self.process_btn.configure(text=" ✨ Traiter Documents", width=self.sidebar_width_expanded - 10)
            self.reset_btn.configure(text=" 🔄 Tout Réinitialiser", width=self.sidebar_width_expanded - 10)
            self.history_btn.configure(text=" 📄 Historique", width=self.sidebar_width_expanded - 10)
            self.help_btn.configure(text=" ❓ Aide", width=self.sidebar_width_expanded - 10)
            self.settings_btn.configure(text=" ⚙️ Paramètres", width=self.sidebar_width_expanded - 10)


        self.sidebar_expanded = not self.sidebar_expanded
        self.app.update_idletasks()


    def setup_chat_area(self):
        # Conteneur pour le chat et l'entrée - Utiliser une couleur de fond
        # On utilise grid pour organiser le textbox et la scrollbar à l'intérieur de ce conteneur.
        chat_container = ctk.CTkFrame(self.conversation_frame, fg_color="#b4d1e8", corner_radius=10) # Couleur de fond du chat demandée
        chat_container.pack(fill="both", expand=True, padx=0, pady=0)

        # Configurer le grid pour le chat_container
        chat_container.grid_columnconfigure(0, weight=1) # La colonne 0 (pour le textbox) s'étire horizontalement
        chat_container.grid_rowconfigure(0, weight=1) # La ligne 0 (pour le textbox et la scrollbar) s'étire verticalement

        # Zone d'affichage des messages (CTkTextbox) - Utiliser une couleur de fond
        # Placer dans la grille, ligne 0, colonne 0, s'étire dans toutes les directions (North-South-East-West)
        self.chat_display = ctk.CTkTextbox(chat_container, wrap="word", activate_scrollbars=False, fg_color="#b4d1e8", text_color="#333333") # Fond, texte sombre pour lisibilité sur fond clair
        self.chat_display.grid(row=0, column=0, sticky="nswe", padx=(10, 0), pady=(10, 5)) # PadX ajusté pour laisser la place à la scrollbar
        self.chat_display.configure(state="disabled")

        # --- Configurer les tags pour le style du texte en utilisant SEULEMENT les couleurs ---
        # La taille de police et le style gras/italique ne peuvent PAS être définis par tag_config pour CTkTextbox.
        # La taille moyenne de l'assistant est gérée par la police de base du widget ou du thème.
        # Les couleurs sont ajustées pour un fond clair
        self.chat_display.tag_config("sender_you", foreground="#00008B") # Bleu foncé pour l'utilisateur
        self.chat_display.tag_config("sender_assistant", foreground="#006400") # Vert foncé pour l'assistant
        self.chat_display.tag_config("user_message", foreground="#333333") # Texte utilisateur sombre
        self.chat_display.tag_config("assistant_message", foreground="#333333") # Texte de l'assistant sombre par défaut

        # Couleurs pour simuler le style Markdown sur fond clair
        self.chat_display.tag_config("bold", foreground="#8B0000") # Rouge foncé pour le gras
        self.chat_display.tag_config("italic", foreground="#800080") # Violet pour l'italique
        self.chat_display.tag_config("bold_italic", foreground="#4B0082") # Indigo pour gras/italique
        self.chat_display.tag_config("h1", foreground="#0000CD") # Bleu moyen pour H1
        self.chat_display.tag_config("h2", foreground="#4682B4") # Gris bleuté pour H2
        self.chat_display.tag_config("h3", foreground="#6A5ACD") # Bleu ardoise pour H3
        self.chat_display.tag_config("typing_indicator", foreground="#555555") # Gris foncé pour l'indicateur


        # Scrollbar pour le chat_display
        # Placer dans la grille, ligne 0, colonne 1, s'étire verticalement (North-South)
        chat_scrollbar = ctk.CTkScrollbar(chat_container, command=self.chat_display.yview, fg_color="#d0d3d6") # Couleur de la scrollbar ajustée
        chat_scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, 10), pady=(10, 5)) # PadY doit correspondre au textbox pour alignement
        self.chat_display.configure(yscrollcommand=chat_scrollbar.set)

        # Zone d'entrée de texte et bouton d'envoi - Utilise pack pour cette ligne sous la zone de chat
        # Placer dans la grille, ligne 1, s'étend sur 2 colonnes, s'étire horizontalement (East-West)
        input_container = ctk.CTkFrame(chat_container, fg_color="transparent")
        input_container.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(5, 10)) # Span across both columns, fill east-west

        self.question_entry = ctk.CTkEntry(input_container, placeholder_text="Ask a question...", fg_color="#f0f3f6", text_color="#333333", border_color="#d0d3d6", corner_radius=20) # Fond, texte, bordure ajustés
        self.question_entry.pack(side="left", fill="x", expand=True, padx=(0, 10)) # Utilise pack à l'intérieur de l'input_container

        self.send_btn = ctk.CTkButton(
            input_container,
            text="Send",
            width=100,
            command=self.ask_question,
            fg_color="#f0f3f6",
            hover_color="#d0d3d6",
            text_color="#333333",
            corner_radius=20
        )
        self.send_btn.pack(side="right") # Utilise pack à l'intérieur de l'input_container

        # Lier la touche Entrée à la méthode ask_question sur le widget d'entrée
        self.question_entry.bind("<Return>", lambda event: self.ask_question())

    def setup_status_bar(self):
        # Barre de statut - Conserve le gris clair
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ctk.CTkLabel(self.status_frame, textvariable=self.status_var, text_color="#333333")
        self.status_label.pack(side="left", padx=10)

        self.progress_bar = ctk.CTkProgressBar(self.status_frame, width=100, fg_color="#cccccc", progress_color="#666666") # Couleurs ajustées pour fond clair
        self.progress_bar.pack(side="right", padx=10)
        self.progress_bar.set(0)
        self.progress_bar.pack_forget()

    # --- Méthodes pour la fenêtre des paramètres ---
    def show_settings(self):
        """Crée et affiche la fenêtre des paramètres"""
        if self.settings_window is None or not self.settings_window.winfo_exists():
            self.settings_window = ctk.CTkToplevel(self.app)
            self.settings_window.title("Settings")
            self.settings_window.geometry("400x400")
            self.settings_window.transient(self.app)
            self.settings_window.grab_set()

            # Appliquer le thème clair/couleur de fond à la fenêtre Toplevel
            # Utiliser la couleur de fond principale demandée
            self.settings_window.configure(fg_color="#b4d1e8")


            # Configuration du contenu de la fenêtre des paramètres
            settings_frame = ctk.CTkFrame(self.settings_window, fg_color="transparent") # Transparent pour utiliser le fg_color de la fenêtre parente
            settings_frame.pack(pady=20, padx=20, fill="both", expand=True)

            settings_title = ctk.CTkLabel(settings_frame, text="Processing and Model Settings", font=ctk.CTkFont(size=16, weight="bold"), text_color="#333333") # Texte sombre
            settings_title.pack(pady=(0, 20))

            # --- Sliders pour les paramètres de traitement ---
            # Les sliders et labels utiliseront les couleurs par défaut du thème Light ou seront ajustés si nécessaire
            self._create_setting_slider(settings_frame, "Chunk Size:", self.chunk_size, 100, 3000, 290, "settings_chunk_size", text_color="#333333", slider_fg_color="#cccccc", slider_progress_color="#666666")
            self._create_setting_slider(settings_frame, "Overlap:", self.chunk_overlap, 0, 1000, 100, "settings_chunk_overlap", text_color="#333333", slider_fg_color="#cccccc", slider_progress_color="#666666")
            self._create_setting_slider(settings_frame, "Results (k):", self.search_k, 1, 10, 9, "settings_search_k", text_color="#333333", slider_fg_color="#cccccc", slider_progress_color="#666666")
            self._create_setting_slider(settings_frame, "Temperature:", self.temperature, 0.0, 1.0, 100, "settings_temperature", is_float=True, text_color="#333333", slider_fg_color="#cccccc", slider_progress_color="#666666")

            # Bouton pour sauvegarder les paramètres
            save_button = ctk.CTkButton(settings_frame, text="Save Settings", command=self.save_settings, fg_color="#f0f3f6", hover_color="#d0d3d6", text_color="#333333", corner_radius=20)
            save_button.pack(pady=20)

        else:
            self.settings_window.focus()

    def _create_setting_slider(self, parent_frame, label_text, current_value, from_, to, steps, attribute_prefix, is_float=False, text_color="white", slider_fg_color=None, slider_progress_color=None):
        """Helper function to create a labeled slider for settings"""
        container = ctk.CTkFrame(parent_frame, fg_color="transparent")
        container.pack(fill="x", pady=5)

        ctk.CTkLabel(container, text=label_text, text_color=text_color).pack(side="left", padx=5)

        var_type = ctk.DoubleVar if is_float else ctk.IntVar
        slider_var = var_type(value=current_value)
        slider_attr_name = f"{attribute_prefix}_var"
        setattr(self, slider_attr_name, slider_var)

        value_label = ctk.CTkLabel(container, text=f"{current_value:.2f}" if is_float else f"{int(current_value)}", width=50, text_color=text_color)
        label_attr_name = f"{attribute_prefix}_label"
        setattr(self, label_attr_name, value_label)

        def update_label(val):
            label_text = f"{val:.2f}" if is_float else f"{int(val)}"
            value_label.configure(text=label_text)

        slider_kwargs = {
            "from_": from_, "to": to, "number_of_steps": steps,
            "variable": slider_var, "command": update_label,
        }
        if slider_fg_color:
             slider_kwargs["fg_color"] = slider_fg_color
        if slider_progress_color:
             slider_kwargs["progress_color"] = slider_progress_color

        slider = ctk.CTkSlider(container, **slider_kwargs)
        slider.pack(side="left", fill="x", expand=True, padx=5)

        value_label.pack(side="right", padx=5)


    def save_settings(self):
        """Sauvegarde les paramètres depuis la fenêtre des paramètres"""
        self.chunk_size = self.settings_chunk_size_var.get()
        self.chunk_overlap = self.settings_chunk_overlap_var.get()
        self.search_k = self.settings_search_k_var.get()
        self.temperature = self.settings_temperature_var.get()

        self.show_message("Settings saved.")
        if self.settings_window is not None:
            self.settings_window.destroy()
            self.settings_window = None

    # --- Méthodes de traitement et RAG ---

    def get_document_text(self, file_paths):
        """
        Extrait le texte de différents types de fichiers (PDF, CSV, TXT, etc.)
        NOTE IMPORTANTE : L'extraction de texte/informations à partir d'images (JPG, PNG, etc.)
        n'est PAS implémentée dans cette méthode car cela nécessite des bibliothèques d'OCR
        ou des modèles multimodaux qui ne sont pas intégrés ici.
        """
        text = ""
        for file_path in file_paths:
            try:
                file_ext = os.path.splitext(file_path)[1].lower()
                
                if file_ext == '.pdf':
                    pdf_reader = PdfReader(file_path)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                            
                elif file_ext == '.csv':
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as csvfile:
                        csv_reader = csv.reader(csvfile)
                        for row in csv_reader:
                            row_text = ', '.join(str(cell) for cell in row)
                            text += row_text + "\n"
                            
                elif file_ext in ['.txt', '.md', '.rst', '.json']:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                        file_text = file.read()
                        text += file_text + "\n\n"
                
                # --- Gestion des types de fichiers image (ajoutés au dialogue, mais pas traités ici) ---
                elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                    print(f"Fichier image sélectionné : {file_path}. L'extraction de texte/informations à partir d'images n'est pas supportée dans cette version du RAG textuel.")
                    # Ne pas ajouter de texte pour les images, continuer simplement
                    continue
                # --- Fin de la gestion des types de fichiers image ---

                else:
                    print(f"Format de fichier non supporté pour l'extraction de texte : {file_ext} pour {file_path}")
                    # Ne pas ajouter de texte, continuer simplement
                    continue
                    
            except Exception as e:
                print(f"Erreur lors de la lecture du fichier {file_path}: {str(e)}")
                traceback.print_exc()
                self.show_message(f"Error reading file {os.path.basename(file_path)}: {str(e)}")
        
        return text


    def get_text_chunks(self, text, chunk_size, chunk_overlap):
        """Découpe le texte en chunks"""
        if not text or not text.strip():
            print("Warning: Extracted text is empty. Cannot create chunks.")
            return []
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks
    
    def get_vectorstore(self, text_chunks):
        """Crée la base vectorielle pour la recherche"""
        if not text_chunks:
            print("Warning: No text chunks available. Cannot create vector store.")
            return None

        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'} # Ajustez si vous avez un GPU
            )
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            return vectorstore
        except Exception as e:
            print(f"Error creating vector store: {e}")
            traceback.print_exc()
            self.show_message(f"Error creating vector store: {str(e)}")
            return None
    
    def get_conversation_chain(self, vectorstore, search_k):
        """Crée la chaîne de conversation"""
        if vectorstore is None:
            print("Warning: Vector store is None. Cannot create conversation chain.")
            return None

        if not API_KEY:
             print("Error: OPENROUTER_API_KEY is not set. Cannot initialize language model.")
             self.show_message("Error: API key not set. Check your .env file.")
             return None

        try:
            llm = ChatOpenAI(
                api_key=API_KEY,
                base_url=API_BASE_URL,
                model_name=MODEL_NAME,
                temperature=self.temperature # Utiliser la température sauvegardée
            )
            
            memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
            
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": search_k}),
                memory=memory
            )
            return conversation_chain
        except Exception as e:
            print(f"Error creating conversation chain: {e}")
            traceback.print_exc()
            self.show_message(f"Error initializing AI: {str(e)}")
            return None

    # --- Méthodes de gestion des fichiers et traitement ---

    def select_files(self):
        """Ouvre un dialogue pour sélectionner différents types de fichiers"""
        files = filedialog.askopenfilenames(
            title="Select documents",
            filetypes=[
                ("Supported documents (PDF, TXT, CSV, MD, RST, JSON)", "*.pdf;*.txt;*.csv;*.md;*.rst;*.json"),
                 # Ajout des types de fichiers image - Notez qu'ils ne sont pas traités ici pour le RAG
                ("Image files (Not processed for RAG)", "*.png;*.jpg;*.jpeg;*.gif;*.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if files:
            # Séparer les fichiers supportés pour le RAG textuel des autres
            supported_files = [f for f in files if os.path.splitext(f)[1].lower() in ['.pdf', '.txt', '.csv', '.md', '.rst', '.json']]
            unsupported_files = [f for f in files if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']] # Exemple de fichiers images

            self.uploaded_files = supported_files # Ne garder que les fichiers supportés pour le traitement RAG textuel
            
            if unsupported_files:
                 unsupported_names = [os.path.basename(f) for f in unsupported_files]
                 msg = f"{len(supported_files)} supported document(s) selected. Image file(s) ({', '.join(unsupported_names)}) were not added for RAG processing as image extraction is not supported in this version."
                 self.show_message(msg)
                 print(msg) # Aussi dans la console

            if supported_files:
                 msg = f"{len(supported_files)} supported document(s) selected."
                 self.show_message(msg)
                 print(msg) # Aussi dans la console

            # --- Note: Mise à jour de l'affichage de la liste des fichiers ---
            # Vous devrez adapter cette partie pour afficher la liste des fichiers
            # dans le nouvel emplacement (ex: fenêtre de paramètres ou section dédiée)
            print("Selected supported documents:")
            for i, file in enumerate(self.uploaded_files):
                 print(f"{i+1}. {os.path.basename(file)}")
            # Si vous avez un widget pour la liste ailleurs, mettez à jour ici
            # ---------------------------------------------------------------

    def process_files(self):
        """Traite les fichiers sélectionnés"""
        if not self.uploaded_files:
            self.show_message("No supported files selected for processing.")
            return
        
        if self.processing:
            self.show_message("Processing in progress, please wait...")
            return
        
        # Désactiver les boutons pendant le traitement
        self.processing = True
        self._disable_buttons()
        self.status_var.set("Processing documents...")
        self.progress_bar.pack(side="right", padx=10)
        
        # Lancer le traitement dans un thread séparé
        threading.Thread(target=self._process_files_thread, daemon=True).start()
    
    def _process_files_thread(self):
        """Thread pour le traitement des fichiers"""
        try:
            self.app.after(0, lambda: self._start_progress_animation())
            
            chunk_size = self.chunk_size
            chunk_overlap = self.chunk_overlap
            search_k = self.search_k

            self.update_status("Extracting text from documents...")
            raw_text = self.get_document_text(self.uploaded_files)
            if not raw_text or not raw_text.strip():
                self.show_message("Could not extract text from documents or extracted text is empty. Processing stopped.")
                return
                        
            self.update_status("Creating text chunks...")
            text_chunks = self.get_text_chunks(raw_text, chunk_size, chunk_overlap)
            if not text_chunks:
                self.show_message("No text chunks could be created. Processing stopped.")
                return
                        
            self.update_status("Creating vector database...")
            vectorstore = self.get_vectorstore(text_chunks)
            if vectorstore is None:
                 # get_vectorstore affiche déjà une erreur si ça échoue
                 return
            
            self.update_status("Initializing conversation chain...")
            self.conversation_chain = self.get_conversation_chain(vectorstore, search_k)
            if self.conversation_chain is None:
                 # get_conversation_chain affiche déjà une erreur si ça échoue
                 return
                        
            self.show_message(f"Processing complete! {len(text_chunks)} chunks created.")
            # Message de bienvenue mis à jour pour refléter le traitement de DOCUMENTS
            self.add_to_chat("Assistant", "**Bonjour !** Je suis prêt à répondre à vos questions sur vos **documents traités**.") # Message mis à jour
            # Ajout d'une note claire sur les images si des images ont été sélectionnées
            # Note: Cette vérification simple suppose que si des images ont été sélectionnées via le dialogue,
            # l'utilisateur s'attend à une note. L'état exact des fichiers sélectionnés est dans self.uploaded_files,
            # mais si des images ont été filtrées, elles ne sont plus dans cette liste.
            # Une vérification plus précise nécessiterait de garder une liste des "tous" les fichiers sélectionnés initialement.
            # Pour simplifier ici, on utilise filedialog à nouveau (potentiellement moins précis si l'utilisateur change de sélection)
            # ou on pourrait ajouter un attribut self.all_selected_files dans select_files.
            try:
                # Tentative de voir si des fichiers image *ont été potentiellement* sélectionnés
                # C'est une approximation, car l'utilisateur pourrait sélectionner des fichiers et les annuler ensuite.
                # Une meilleure approche serait de stocker le résultat de filedialog.askopenfilenames dans un autre attribut.
                # Pour ce code complet, utilisons une simple vérification si self.uploaded_files n'est pas vide
                # (indiquant qu'au moins des documents SUPPORTÉS ont été sélectionnés et potentiellement aussi des images).
                # On pourrait aussi ajouter un booléen self.image_files_were_selected dans select_files.
                # Simplifions : si des documents supportés ont été traités, on affiche la note d'image
                # pour couvrir le cas où des images *ont été essayées* d'être uploadées avec les docs.
                 if self.uploaded_files: # Si des documents supportés ont été traités
                     self.add_to_chat("Assistant", "Veuillez noter que les images sélectionnées ne sont pas traitées pour le RAG dans cette version.")
            except Exception as e:
                 print(f"Error checking for image files: {e}")
                 pass # Continuer même si la vérification échoue


        except Exception as e:
            print(f"Error during file processing thread: {str(e)}")
            traceback.print_exc()
            self.show_message(f"Error during processing: {str(e)}")
            self.conversation_chain = None # S'assurer que la chaîne est None en cas d'erreur
        
        finally:
            # Réactiver les boutons
            self.app.after(0, lambda: self._enable_buttons())

    # --- Méthodes de conversation ---

    def ask_question(self):
        """Traite la question de l'utilisateur"""
        question = self.question_entry.get().strip()
        if not question:
            return
            
        if self.processing: # Ne pas permettre de poser une question pendant le traitement des fichiers
             self.show_message("Please wait for document processing to complete.")
             return

        if self.conversation_chain is None:
            self.show_message("Please process documents first.")
            return
        
        # Vider le champ de saisie
        self.question_entry.delete(0, tk.END)
        
        # Afficher la question dans le chat
        self.add_to_chat("Moi", question)
        
        # Désactiver les boutons pendant le traitement
        self.send_btn.configure(state="disabled")
        self.question_entry.configure(state="disabled")
        self.status_var.set("Thinking...")
        
        # Traiter la question dans un thread séparé
        threading.Thread(target=self._process_question_thread, args=(question,), daemon=True).start()

    def _process_question_thread(self, question):
        """Thread pour le traitement de la question"""
        try:
            # Mettre à jour le statut pour informer l'utilisateur
            self.app.after(0, lambda: self.status_var.set("Thinking..."))
            
            # Afficher un indicateur de chargement dans le chat (optionnel)
            self.app.after(0, lambda: self._show_typing_indicator(True))
            
            # Obtenir une réponse
            response = self.conversation_chain({"question": question})
            answer = response.get("answer", "Sorry, I couldn't find an answer to your question based on the documents.")
            
            # Masquer l'indicateur de chargement
            self.app.after(0, lambda: self._show_typing_indicator(False))
            
            # add_to_chat gérera l'affichage stylisé
            self.app.after(0, lambda: self.add_to_chat("Mariem", answer))
            
            # Enregistrer cette interaction dans l'historique
            self.app.after(0, lambda: self.save_history_entry(question, answer, self.uploaded_files))
            
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            traceback.print_exc()
            self.app.after(0, lambda: self.show_message(f"Error processing question: {str(e)}"))
            self.app.after(0, lambda: self._show_typing_indicator(False))
            
        finally:
            # Réactiver les boutons
            self.app.after(0, lambda: self.send_btn.configure(state="normal"))
            self.app.after(0, lambda: self.question_entry.configure(state="normal"))
            self.app.after(0, lambda: self.status_var.set("Ready"))

    # --- Méthodes utilitaires pour l'interface et le statut ---

    def update_status(self, message):
        """Met à jour le texte de la barre de statut"""
        self.app.after(0, lambda: self.status_var.set(message))

    def show_message(self, message):
        """Affiche un message informatif ou d'erreur dans la barre de statut et la console"""
        print(message)
        self.update_status(message)

    def _disable_buttons(self):
        """Désactive les boutons pertinents pendant le traitement"""
        buttons_to_disable = [
            self.upload_btn, self.process_btn, self.send_btn, self.question_entry,
            self.history_btn, self.help_btn, self.settings_btn, self.new_chat_btn
        ]
        for btn in buttons_to_disable:
            if isinstance(btn, (ctk.CTkButton, ctk.CTkEntry)):
                btn.configure(state="disabled")

    def _enable_buttons(self):
        """Réactive les boutons après le traitement"""
        buttons_to_enable = [
            self.upload_btn, self.process_btn, self.send_btn, self.question_entry,
            self.history_btn, self.help_btn, self.settings_btn, self.new_chat_btn
        ]
        for btn in buttons_to_enable:
             if isinstance(btn, (ctk.CTkButton, ctk.CTkEntry)):
                btn.configure(state="normal")


        self.processing = False
        self.status_var.set("Ready")
        self.progress_bar.set(1.0)
        # Attendre un peu avant de masquer la barre de progression
        self.app.after(500, lambda: self.progress_bar.pack_forget())

    def _start_progress_animation(self):
        """Démarre l'animation de la barre de progression"""
        self.progress_bar.set(0)
        self.progress_bar.pack(side="right", padx=10) # S'assurer qu'elle est visible
        self._update_progress_bar(0.05)

    def _update_progress_bar(self, increment):
        """Met à jour la barre de progression avec animation"""
        current = self.progress_bar.get()
        if self.processing:
            if current < 0.9: # Arrêter un peu avant la fin pour la sensation de chargement
                self.progress_bar.set(current + increment)
                # Diminuer légèrement l'incrément pour ralentir vers la fin
                self.app.after(200, lambda: self._update_progress_bar(increment * 0.98))
            # else: Si la barre atteint 90% et que processing est toujours True, elle attend la fin du traitement.
        else:
            # Quand processing devient False, compléter la barre et la masquer
            self.progress_bar.set(1.0)
            self.app.after(500, lambda: self.progress_bar.pack_forget())


    def _show_typing_indicator(self, show=True):
        """Affiche ou masque un indicateur que l'assistant est en train de taper"""
        if not hasattr(self, 'chat_display') or not self.chat_display.winfo_exists():
            return

        self.chat_display.configure(state="normal")
        typing_indicator_tag = "typing_indicator"
        indicator_text = "🤖 typing..."

        # Trouver la position actuelle de l'indicateur s'il existe
        indicator_start = None
        indicator_end = None
        # On cherche si le tag 'typing_indicator' est appliqué quelque part
        ranges = self.chat_display.tag_ranges(typing_indicator_tag)
        if ranges:
            # S'il y a des ranges, l'indicateur existe. On prend le dernier range.
            indicator_start = ranges[-2] # Indices viennent par paires (start, end, start, end, ...)
            indicator_end = ranges[-1]


        if show:
            if indicator_start is None: # Si l'indicateur n'est pas déjà là
                 # Insérer un espace si nécessaire avant l'indicateur
                if self.chat_display.get("1.0", "end-1c").strip():
                     self.chat_display.insert(tk.END, "\n\n")
                
                # Insérer l'indicateur avec le tag
                self.chat_display.insert(tk.END, indicator_text, typing_indicator_tag)
                self.chat_display.see(tk.END) # Scroll to the end
        else:
            if indicator_start is not None: # Si l'indicateur est là et qu'on doit le cacher
                try:
                    self.chat_display.delete(indicator_start, indicator_end)
                    # Optionnel: Supprimer la ligne vide précédente si elle a été ajoutée spécifiquement pour l'indicateur
                    # Cela nécessite une logique plus complexe pour être sûr de ne supprimer que la ligne vide ajoutée par _show_typing_indicator
                    # Pour l'instant, on ignore cette complexité.
                except Exception as e:
                     print(f"Error deleting typing indicator: {e}")
                     pass # Gérer l'erreur si la suppression échoue


        self.chat_display.configure(state="disabled")
        self.chat_display.see(tk.END)


    def add_to_chat(self, sender, message):
        """Ajoute un message au chat avec avatars, timestamps et styles"""
        self.chat_display.configure(state="normal")
        timestamp = time.strftime("%H:%M:%S")
        
        # Insérer un séparateur si le chat n'est pas vide
        if self.chat_display.get("1.0", tk.END).strip():
            self.chat_display.insert(tk.END, "\n\n")
        
        # Avatar et nom de l'expéditeur
        if sender == "You":
            avatar = "👤"
            self.chat_display.insert(tk.END, f"{avatar} [{timestamp}] ", "sender_you")
            self.chat_display.insert(tk.END, "**You**\n", "sender_you_name") # Tag spécifique pour le nom si besoin de style différent

            # Pour les messages de l'utilisateur, pas de parsing markdown complexe ni d'effet de frappe
            self.chat_display.insert(tk.END, message + "\n", "user_message")
            self.chat_display.see(tk.END)
            self.chat_display.configure(state="disabled")
            
        else: # Assistant
            avatar = "🤖"
            self.chat_display.insert(tk.END, f"{avatar} [{timestamp}] ", "sender_assistant")
            self.chat_display.insert(tk.END, "**Assistant**\n", "sender_assistant_name") # Tag spécifique pour le nom

            self.chat_display.configure(state="disabled") # Désactiver temporairement pour l'effet de frappe stylisé
            
            # Utiliser l'effet de frappe stylisé pour le message de l'assistant
            self._type_effect(message)


    def _type_effect(self, message_content, index=0, current_tags=None):
        """Affiche le texte de l'assistant progressivement avec parsing Markdown (couleurs et uppercase)."""
        if not hasattr(self, 'chat_display') or not self.chat_display.winfo_exists():
            return

        self.chat_display.configure(state="normal")

        if current_tags is None:
            current_tags = ["assistant_message"] # Tag de base pour le message de l'assistant (pour la couleur par défaut)

        if index < len(message_content):
            char = message_content[index]
            remaining_text = message_content[index:]

            # --- Logique de parsing Markdown simple et mise à jour des tags de COULEUR ---
            # Note: Cette logique est simplifiée pour l'effet de frappe caractère par caractère.
            # Une implémentation complète nécessiterait une analyse préalable du texte entier.

            new_tags = [tag for tag in current_tags if tag not in ['bold', 'italic', 'bold_italic']] # Réinitialiser les tags de style pour ce caractère

            # Détecter le début des marqueurs
            if remaining_text.startswith("***") or remaining_text.startswith("___"):
                 # Vérifier si le marqueur de fin existe pour ce début (simple vérification d'existence)
                 end_marker = remaining_text[:3]
                 if remaining_text[3:].find(end_marker) != -1:
                     new_tags.append('bold_italic')
                     self.app.after(random.randint(5, 20), self._type_effect, message_content, index + 3, new_tags)
                     return # Sauter le marqueur et continuer

            elif remaining_text.startswith("**") or remaining_text.startswith("__"):
                 end_marker = remaining_text[:2]
                 if remaining_text[2:].find(end_marker) != -1:
                    new_tags.append('bold')
                    self.app.after(random.randint(5, 20), self._type_effect, message_content, index + 2, new_tags)
                    return # Sauter le marqueur et continuer

            elif remaining_text.startswith("*") and not remaining_text.startswith("***"):
                 if remaining_text[1:].find("*") != -1:
                    new_tags.append('italic')
                    self.app.after(random.randint(5, 20), self._type_effect, message_content, index + 1, new_tags)
                    return # Sauter le marqueur et continuer

            elif remaining_text.startswith("_") and not remaining_text.startswith("___"):
                 if remaining_text[1:].find("_") != -1:
                    new_tags.append('italic')
                    self.app.after(random.randint(5, 20), self._type_effect, message_content, index + 1, new_tags)
                    return # Sauter le marqueur et continuer


            # --- Fin Logique de parsing Markdown (Début) ---

            # Gérer la mise en majuscules pour le texte qui devrait être gras ou italique
            char_to_insert = char
            # Vérification simplifiée si le caractère est potentiellement entouré de marqueurs (approximation)
            # Cette logique peut ne pas être parfaite pour tous les cas imbriqués ou complexes.
            is_potentially_styled = False
            # On vérifie si le caractère actuel est *à l'intérieur* d'une section qui aurait le style.
            # Cette détection simple regarde si les marqueurs *précèdent* et *suivent* ce caractère.
            # C'est une approximation pour la frappe progressive. Une analyse complète du texte avant
            # l'effet de frappe serait plus précise.

            # Vérifier si l'un des tags de style (bold, italic, bold_italic) est actuellement actif pour ce caractère
            if any(tag in ['bold', 'italic', 'bold_italic'] for tag in current_tags):
                 is_potentially_styled = True

            # On peut ajouter ici une logique plus complexe si nécessaire pour détecter les marqueurs de fin juste avant l'index actuel
            # ... (logique de détection de marqueurs de fin ici si nécessaire) ...


            if is_potentially_styled:
                 if char.isalpha():
                     char_to_insert = char.upper()


            # Insérer le caractère avec les tags de couleur actifs
            # S'assurer que le tag de message assistant est toujours présent pour la couleur de base
            final_tags = list(new_tags)
            if "assistant_message" not in final_tags:
                final_tags.append("assistant_message")


            self.chat_display.insert(tk.END, char_to_insert, tuple(final_tags))
            self.chat_display.see(tk.END)

            # Appeler la fonction pour le caractère suivant
            self.app.after(random.randint(10, 40), self._type_effect, message_content, index + 1, new_tags) # Passer les tags potentiellement mis à jour
        else:
            # Fin de l'effet de frappe
            # Appliquer les tags de titre (couleur et uppercase) après que tout le texte est inséré
            self._apply_header_tags()

            # S'assurer qu'il y a un saut de ligne à la fin
            if not self.chat_display.get("end-2c", "end-1c").endswith("\n"):
                 self.chat_display.insert(tk.END, "\n")

            self.chat_display.configure(state="disabled")


    def _apply_header_tags(self):
        """Applique les tags de titre (couleur et uppercase) après que le texte est inséré."""
        content_lines = self.chat_display.get("1.0", "end-1c").splitlines()
        current_pos = "1.0"

        self.chat_display.configure(state="normal")

        for line in content_lines:
            line_start_pos = current_pos
            # Déterminer la fin de la ligne actuelle
            line_end_pos = f"{line_start_pos} + {len(line)}c"

            header_match = re.match(r'^(#+)\s*(.*)', line)
            if header_match:
                hashes = header_match.group(1)
                header_text = header_match.group(2).strip()
                level = len(hashes)

                tag_name = None
                display_text = line # Par défaut, afficher la ligne originale

                if level == 1:
                    tag_name = "h1"
                    display_text = header_text.upper() # Mettre en majuscules le texte du titre
                elif level == 2:
                    tag_name = "h2"
                    display_text = header_text.upper() # Mettre en majuscules
                elif level == 3:
                    tag_name = "h3"
                    display_text = header_text # Pas en majuscules pour h3

                if tag_name:
                    # Supprimer l'ancienne ligne et insérer le nouveau texte stylisé
                    self.chat_display.delete(line_start_pos, line_end_pos)
                    self.chat_display.insert(line_start_pos, display_text, tag_name)
                    # Mettre à jour la position courante basée sur la nouvelle longueur de texte inséré
                    current_pos = f"{line_start_pos} + {len(display_text)}c"
                    # S'assurer qu'il y a un saut de ligne après (si l'original en avait un, il sera réinséré par splitlines/join si on reconstruisait le texte)
                    # Ici, on avance simplement la position d'une ligne.
                    current_pos = f"{current_pos} + 1l" # Passer au début de la ligne suivante

                else:
                     # Si ce n'est pas un titre, avancer simplement d'une ligne pour le prochain tour
                     current_pos = f"{line_end_pos} + 1l"

            else:
                 # Si ce n'est pas un titre, avancer simplement d'une ligne
                 current_pos = f"{line_end_pos} + 1l"

        self.chat_display.configure(state="disabled")


    def reset_conversation(self):
        """Réinitialise la conversation, l'historique, les fichiers, etc."""
        self.conversation_chain = None
        self.history = [] # Vider l'historique en mémoire
        self.uploaded_files = [] # Vider la liste des fichiers téléchargés
        self.session_id = str(uuid.uuid4()) # Générer un nouvel ID de session

        # Vider l'affichage du chat
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.configure(state="disabled")

        # Note : Si vous affichez la liste des fichiers ou l'historique ailleurs, mettez à jour ces affichages ici.

        self.show_message("Conversation, files, and history reset. Ready for a new chat.")

    # --- Méthodes de gestion de l'historique ---

    def save_history_entry(self, question, answer, document_names):
        """Sauvegarde une entrée dans l'historique"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Créer une entrée d'historique
        history_entry = {
            "timestamp": timestamp,
            "documents": [os.path.basename(doc) for doc in document_names],
            "question": question,
            "answer": answer
        }
        
        # Ajouter à l'historique en mémoire
        self.history.append(history_entry)
        
        # Limiter la taille de l'historique en mémoire (optionnel)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        # Sauvegarder dans un fichier JSON spécifique à la session
        # Vous pourriez aussi vouloir sauvegarder un historique global
        history_file = self.history_folder / f"history_{self.session_id}.json"
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'historique: {str(e)}")
            traceback.print_exc()

        # Mettre à jour l'affichage de l'historique si implémenté
        # self.update_history_display()

    def load_history(self):
        """Charge les entrées d'historique précédentes (charge le plus récent fichier de la session en cours)"""
        history_file = self.history_folder / f"history_{self.session_id}.json"
        if not history_file.exists():
            print(f"No history file found for session {self.session_id}. Starting fresh.")
            self.history = []
            return

        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                self.history = json.load(f)
            print(f"Loaded {len(self.history)} history entries for session {self.session_id}.")
            
            # Charger la conversation dans la mémoire LangChain si la chaîne existe
            if self.conversation_chain and hasattr(self.conversation_chain, 'memory'):
                # C'est une approche simplifiée, recharger l'historique exact dans LangChain peut être complexe
                # en dehors de la structure de la chaîne. Une alternative est de recréer la mémoire
                # et de la passer à une nouvelle chaîne si nécessaire.
                # Pour l'instant, on charge juste l'historique en mémoire de l'app.
                print("Note: LangChain conversation memory is not automatically reloaded from history file.")
                pass

            # Mettre à jour l'affichage de l'historique si implémenté
            # self.update_history_display()

        except Exception as e:
            print(f"Erreur lors du chargement de l'historique pour la session {self.session_id}: {str(e)}")
            traceback.print_exc()
            self.history = [] # Vider l'historique en cas d'erreur de chargement

    def show_history(self):
        """Affiche l'historique de conversation (implémentation simple)"""
        # Pour une interface plus complète, vous créeriez une nouvelle fenêtre
        # ou un cadre dans l'UI principale pour afficher l'historique formaté.
        print("\n--- Historique de la conversation ---")
        if not self.history:
            print("Aucune entrée d'historique.")
            self.show_message("No history entries.")
            return

        # Afficher dans la console pour l'exemple
        for entry in self.history:
            print(f"[{entry.get('timestamp', 'N/A')}] Documents: {', '.join(entry.get('documents', []))}")
            print(f"Q: {entry.get('question', 'N/A')}")
            # Afficher les premières lignes de la réponse pour ne pas surcharger la console
            answer_preview = entry.get('answer', 'N/A')
            print(f"A: {answer_preview[:100]}...")
            print("-" * 20)
        print("------------------------------------")

        # Si vous avez un widget dédié, mettez-le à jour ici
        # self.update_history_display()
        self.show_message("History printed to console.")


    # --- Méthodes d'aide (Placeholder) ---
    def show_help(self):
        """Affiche les informations d'aide (Placeholder)"""
        # Implémentez l'affichage dans une fenêtre ou section de l'UI ici
        print("\n--- Aide ---")
        print("DocChat AI: Ask questions about your documents.")
        print("1. Use 'Uploader Fichiers' to select PDF, TXT, CSV, MD, RST, or JSON files.")
        print("   Image files can be selected but are NOT processed for RAG in this version.")
        print("2. Use 'Traiter Documents' to process the selected supported files and prepare the AI.")
        print("3. Type your questions in the input box and press Enter or Send.")
        print("4. Use 'Paramètres' to adjust chunking and model temperature settings.")
        print("5. Use 'Historique' to view past interactions (currently in console).")
        print("6. Use 'Tout Réinitialiser' to clear the current session.")
        print("---")
        self.show_message("Help info printed to console.")


    # --- Méthode principale ---
    def run(self):
        """Lance l'application"""
        self.app.mainloop()

# --- Bloc d'exécution principal ---
if __name__ == "__main__":
    app = ModernRAGApp()
    app.run()
