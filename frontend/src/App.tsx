
import { useCallback, useEffect, useRef, useState } from "react"
import axios from "axios"
import {
  MessageSquare,
  Files,
  Settings,
  Plus,
  PanelLeftClose,
  PanelLeftOpen,
  ArrowUp,
  FileText,
  Search,
  MoreHorizontal,
  Upload,
  LoaderCircle,
  CheckCircle2,
  AlertCircle,
  ChevronDown,
  Trash2,
  Pencil,
  Copy,
  X,
  MessageCircle,
  Check,
} from "lucide-react"

const API_URL = "http://127.0.0.1:8000"

type Page = "chat" | "documents"

interface DocumentItem {
  document_id: number
  filename: string
  status: string
  attempt_count: number
  last_error: string | null
}

interface SourceItem {
  chunk_id: string
  document_id: number
  chunk_index: number
  distance: number
  content: string
}

interface ChatMessage {
  id: number
  role: "user" | "assistant"
  content: string
  sources?: SourceItem[]
  error?: boolean
  created_at?: string
}

interface ConversationItem {
  conversation_id: number
  title: string
  document_id: number | null
  created_at: string
  updated_at: string
  message_count: number
}

interface ConversationDetails {
  conversation_id: number
  title: string
  document_id: number | null
  created_at: string
  updated_at: string
  messages: {
    message_id: number
    role: "user" | "assistant"
    content: string
    sources: SourceItem[]
    created_at: string
  }[]
}

interface AskResponse {
  query: string
  answer: string
  sources: SourceItem[]
}

function App() {
  const [activePage, setActivePage] = useState<Page>("chat")
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // Documents
  const [documents, setDocuments] = useState<DocumentItem[]>([])
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [documentsError, setDocumentsError] = useState<string | null>(null)
  const [documentSearch, setDocumentSearch] = useState("")
  const [documentMenuId, setDocumentMenuId] = useState<number | null>(null)
  const [selectedDocumentDetails, setSelectedDocumentDetails] =
    useState<DocumentItem | null>(null)
  const [deletingDocumentId, setDeletingDocumentId] = useState<number | null>(
    null,
  )
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Conversations
  const [conversations, setConversations] = useState<ConversationItem[]>([])
  const [conversationSearch, setConversationSearch] = useState("")
  const [activeConversationId, setActiveConversationId] = useState<number | null>(
    null,
  )
  const [loadingConversations, setLoadingConversations] = useState(false)
  const [loadingConversation, setLoadingConversation] = useState(false)
  const [conversationMenuId, setConversationMenuId] = useState<number | null>(
    null,
  )
  const [renamingConversationId, setRenamingConversationId] = useState<
    number | null
  >(null)
  const [renameValue, setRenameValue] = useState("")
  const [deletingConversationId, setDeletingConversationId] = useState<
    number | null
  >(null)

  // Chat
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [chatInput, setChatInput] = useState("")
  const [sendingMessage, setSendingMessage] = useState(false)
  const [selectedDocumentId, setSelectedDocumentId] = useState("all")
  const chatEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const [notice, setNotice] = useState<string | null>(null)

  const showNotice = (message: string) => {
    setNotice(message)
    window.setTimeout(() => setNotice(null), 3000)
  }

  // ---------------------------------------------------------------------------
  // Documents
  // ---------------------------------------------------------------------------

  const fetchDocuments = useCallback(async () => {
    try {
      const response = await axios.get<DocumentItem[]>(
        `${API_URL}/documents/`,
      )
      setDocuments(response.data)
      setDocumentsError(null)
    } catch (error) {
      console.error("Failed to load documents:", error)
      setDocumentsError("Failed to load documents from the server.")
    }
  }, [])

  useEffect(() => {
    fetchDocuments()
  }, [fetchDocuments])

  const handleUpload = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0]
    if (!file) return

    setUploadError(null)

    const extension = file.name.split(".").pop()?.toLowerCase()
    if (!["pdf", "txt"].includes(extension || "")) {
      setUploadError("Only PDF and TXT files are supported.")
      event.target.value = ""
      return
    }

    const formData = new FormData()
    formData.append("file", file)
    setUploading(true)

    try {
      await axios.post(`${API_URL}/documents/upload`, formData)
      await fetchDocuments()
      showNotice("Document uploaded successfully.")
    } catch (error) {
      if (axios.isAxiosError(error)) {
        const detail = error.response?.data?.detail
        setUploadError(
          typeof detail === "string"
            ? detail
            : error.message || "Upload failed.",
        )
      } else {
        setUploadError("Something went wrong while uploading.")
      }
    } finally {
      setUploading(false)
      event.target.value = ""
    }
  }

  const hasProcessingDocuments = documents.some(
    (document) =>
      !["ready", "completed", "failed"].includes(
        document.status.toLowerCase(),
      ),
  )

  useEffect(() => {
    if (!hasProcessingDocuments) return

    const interval = window.setInterval(fetchDocuments, 3000)
    return () => window.clearInterval(interval)
  }, [hasProcessingDocuments, fetchDocuments])

  const deleteDocument = async (document: DocumentItem) => {
    const confirmed = window.confirm(
      `Delete "${document.filename}"?\n\nThis will remove the document, its database records, and its indexed vectors.`,
    )

    if (!confirmed) return

    setDeletingDocumentId(document.document_id)
    setDocumentMenuId(null)

    try {
      await axios.delete(`${API_URL}/documents/${document.document_id}`)
      setDocuments((previous) =>
        previous.filter(
          (item) => item.document_id !== document.document_id,
        ),
      )

      if (selectedDocumentId === String(document.document_id)) {
        setSelectedDocumentId("all")
      }

      if (
        selectedDocumentDetails?.document_id === document.document_id
      ) {
        setSelectedDocumentDetails(null)
      }

      showNotice("Document deleted.")
    } catch (error) {
      const detail = axios.isAxiosError(error)
        ? error.response?.data?.detail
        : null

      window.alert(
        typeof detail === "string"
          ? detail
          : "Failed to delete document.",
      )
    } finally {
      setDeletingDocumentId(null)
    }
  }

  const getStatusDisplay = (status: string) => {
    switch (status.toLowerCase()) {
      case "ready":
      case "completed":
        return {
          label: "Ready",
          icon: <CheckCircle2 size={14} />,
          className: "bg-green-50 text-green-700",
        }
      case "failed":
        return {
          label: "Failed",
          icon: <AlertCircle size={14} />,
          className: "bg-red-50 text-red-700",
        }
      default:
        return {
          label: status,
          icon: <LoaderCircle size={14} className="animate-spin" />,
          className: "bg-amber-50 text-amber-700",
        }
    }
  }

  const readyDocuments = documents.filter((document) =>
    ["ready", "completed"].includes(document.status.toLowerCase()),
  )

  const filteredDocuments = documents.filter((document) =>
    document.filename.toLowerCase().includes(documentSearch.toLowerCase()),
  )

  // ---------------------------------------------------------------------------
  // Conversations
  // ---------------------------------------------------------------------------

  const fetchConversations = useCallback(async (query = "") => {
    setLoadingConversations(true)

    try {
      const response = await axios.get<ConversationItem[]>(
        `${API_URL}/conversations/`,
        {
          params: query.trim() ? { q: query.trim() } : undefined,
        },
      )
      setConversations(response.data)
    } catch (error) {
      console.error("Failed to load conversations:", error)
    } finally {
      setLoadingConversations(false)
    }
  }, [])

  useEffect(() => {
    fetchConversations()
  }, [fetchConversations])

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      fetchConversations(conversationSearch)
    }, 250)

    return () => window.clearTimeout(timeout)
  }, [conversationSearch, fetchConversations])

  const createConversation = async () => {
    try {
      const response = await axios.post<ConversationItem>(
        `${API_URL}/conversations/`,
        {
          title: "New chat",
          document_id:
            selectedDocumentId === "all"
              ? null
              : Number(selectedDocumentId),
        },
      )

      const conversation = response.data

      setActiveConversationId(conversation.conversation_id)
      setMessages([])
      setChatInput("")
      setSelectedDocumentId(
        conversation.document_id === null
          ? "all"
          : String(conversation.document_id),
      )
      setActivePage("chat")
      setConversationMenuId(null)

      await fetchConversations()
      textareaRef.current?.focus()
    } catch (error) {
      console.error("Failed to create conversation:", error)
      showNotice("Could not create a new conversation.")
    }
  }

  const openConversation = async (conversationId: number) => {
    if (sendingMessage) return

    setLoadingConversation(true)
    setActivePage("chat")
    setConversationMenuId(null)

    try {
      const response = await axios.get<ConversationDetails>(
        `${API_URL}/conversations/${conversationId}`,
      )

      const conversation = response.data

      setActiveConversationId(conversation.conversation_id)
      setSelectedDocumentId(
        conversation.document_id === null
          ? "all"
          : String(conversation.document_id),
      )

      setMessages(
        conversation.messages.map((message) => ({
          id: message.message_id,
          role: message.role,
          content: message.content,
          sources: message.sources || [],
          created_at: message.created_at,
        })),
      )
    } catch (error) {
      console.error("Failed to load conversation:", error)
      showNotice("Could not load this conversation.")
    } finally {
      setLoadingConversation(false)
    }
  }

  const renameConversation = async (conversationId: number) => {
    const title = renameValue.trim()

    if (!title) {
      setRenamingConversationId(null)
      return
    }

    try {
      await axios.patch(`${API_URL}/conversations/${conversationId}`, {
        title,
      })

      setRenamingConversationId(null)
      setRenameValue("")
      await fetchConversations(conversationSearch)
      showNotice("Conversation renamed.")
    } catch (error) {
      console.error("Failed to rename conversation:", error)
      showNotice("Could not rename conversation.")
    }
  }

  const deleteConversation = async (conversationId: number) => {
    const confirmed = window.confirm(
      "Delete this conversation and all its messages?",
    )

    if (!confirmed) return

    setDeletingConversationId(conversationId)
    setConversationMenuId(null)

    try {
      await axios.delete(`${API_URL}/conversations/${conversationId}`)

      setConversations((previous) =>
        previous.filter(
          (item) => item.conversation_id !== conversationId,
        ),
      )

      if (activeConversationId === conversationId) {
        setActiveConversationId(null)
        setMessages([])
        setChatInput("")
        setSelectedDocumentId("all")
      }

      showNotice("Conversation deleted.")
    } catch (error) {
      console.error("Failed to delete conversation:", error)
      showNotice("Could not delete conversation.")
    } finally {
      setDeletingConversationId(null)
    }
  }

  const startNewConversation = () => {
    if (sendingMessage) return

    setActiveConversationId(null)
    setMessages([])
    setChatInput("")
    setSelectedDocumentId("all")
    setActivePage("chat")
    setConversationMenuId(null)
    textareaRef.current?.focus()
  }

  // ---------------------------------------------------------------------------
  // Chat
  // ---------------------------------------------------------------------------

  const handleSendMessage = async () => {
    const query = chatInput.trim()

    if (!query || sendingMessage) return

    setChatInput("")
    setSendingMessage(true)

    const temporaryId = Date.now()

    setMessages((previous) => [
      ...previous,
      {
        id: temporaryId,
        role: "user",
        content: query,
      },
    ])

    try {
      let conversationId = activeConversationId

      // Create a persistent conversation before sending its first question.
      if (conversationId === null) {
        const createResponse = await axios.post<ConversationItem>(
          `${API_URL}/conversations/`,
          {
            title: "New chat",
            document_id:
              selectedDocumentId === "all"
                ? null
                : Number(selectedDocumentId),
          },
        )

        conversationId = createResponse.data.conversation_id
        setActiveConversationId(conversationId)
      }

      const response = await axios.post<AskResponse>(
        `${API_URL}/rag/ask`,
        {
          query,
          conversation_id: conversationId,
          document_id:
            selectedDocumentId === "all"
              ? null
              : Number(selectedDocumentId),
          n_results: 5,
        },
      )

      setMessages((previous) => [
        ...previous,
        {
          id: temporaryId + 1,
          role: "assistant",
          content: response.data.answer,
          sources: response.data.sources || [],
        },
      ])

      await fetchConversations(conversationSearch)
    } catch (error) {
      console.error("Failed to send chat message:", error)

      let errorMessage = "Failed to generate an answer. Please try again."

      if (axios.isAxiosError(error)) {
        const detail = error.response?.data?.detail

        if (typeof detail === "string") {
          errorMessage = detail
        } else if (error.message) {
          errorMessage = error.message
        }
      }

      setMessages((previous) => [
        ...previous,
        {
          id: temporaryId + 2,
          role: "assistant",
          content: errorMessage,
          error: true,
        },
      ])
    } finally {
      setSendingMessage(false)
    }
  }

  const handleChatKeyDown = (
    event: React.KeyboardEvent<HTMLTextAreaElement>,
  ) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()
      handleSendMessage()
    }
  }

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({
      behavior: "smooth",
    })
  }, [messages, sendingMessage])

  const activeConversation = conversations.find(
    (conversation) =>
      conversation.conversation_id === activeConversationId,
  )

  // ---------------------------------------------------------------------------
  // Utilities
  // ---------------------------------------------------------------------------

  const copyText = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      showNotice("Copied to clipboard.")
    } catch {
      showNotice("Could not copy to clipboard.")
    }
  }

  const formatDate = (dateString: string) => {
    if (!dateString) return ""

    const date = new Date(dateString)

    if (Number.isNaN(date.getTime())) return ""

    return date.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
    })
  }

  const groupedConversations = (() => {
    const today = new Date()
    const startOfToday = new Date(
      today.getFullYear(),
      today.getMonth(),
      today.getDate(),
    )

    const yesterday = new Date(startOfToday)
    yesterday.setDate(yesterday.getDate() - 1)

    const groups: Record<string, ConversationItem[]> = {
      Today: [],
      Yesterday: [],
      Previous: [],
    }

    conversations.forEach((conversation) => {
      const date = new Date(conversation.updated_at || conversation.created_at)

      if (Number.isNaN(date.getTime())) {
        groups.Previous.push(conversation)
      } else if (date >= startOfToday) {
        groups.Today.push(conversation)
      } else if (date >= yesterday) {
        groups.Yesterday.push(conversation)
      } else {
        groups.Previous.push(conversation)
      }
    })

    return groups
  })()

  return (
    <div className="flex h-screen overflow-hidden bg-[#faf9f6] text-[#292824]">
      {/* Sidebar */}
      <aside
        className={`${
          sidebarOpen ? "w-[270px]" : "w-0"
        } flex shrink-0 flex-col overflow-hidden border-r border-[#e9e7e2] bg-[#f5f4f0] transition-all duration-200`}
      >
        <div className="flex h-14 shrink-0 items-center justify-between px-5">
          <h1 className="text-xl font-semibold tracking-tight">
            LexiCore<span className="text-[#b58b58]">.</span>
          </h1>

          <button
            onClick={() => setSidebarOpen(false)}
            className="rounded-md p-1.5 text-gray-500 hover:bg-[#eae8e2] hover:text-gray-800"
            title="Collapse sidebar"
          >
            <PanelLeftClose size={18} />
          </button>
        </div>

        <div className="px-3 pt-4">
          <button
            onClick={startNewConversation}
            className="flex w-full items-center gap-3 rounded-lg bg-[#eae8e2] px-3 py-2.5 text-sm font-medium hover:bg-[#e3e1da]"
          >
            <Plus size={17} />
            New chat
          </button>
        </div>

        <div className="px-3 pt-3">
          <div className="relative">
            <Search
              size={15}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400"
            />
            <input
              value={conversationSearch}
              onChange={(event) => setConversationSearch(event.target.value)}
              placeholder="Search chats"
              className="w-full rounded-lg border border-transparent bg-[#eeede8] py-2 pl-9 pr-3 text-sm outline-none placeholder:text-gray-400 focus:border-[#d7d3ca] focus:bg-white"
            />
          </div>
        </div>

        <div className="px-3 pt-6">
          <p className="mb-2 px-2 text-[11px] font-semibold uppercase tracking-wider text-gray-400">
            Workspace
          </p>

          <nav className="space-y-1">
            <button
              onClick={() => setActivePage("chat")}
              className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm ${
                activePage === "chat"
                  ? "bg-[#eae8e2] font-medium"
                  : "text-gray-600 hover:bg-[#eae8e2]"
              }`}
            >
              <MessageSquare size={16} />
              Chat
            </button>

            <button
              onClick={() => setActivePage("documents")}
              className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm ${
                activePage === "documents"
                  ? "bg-[#eae8e2] font-medium"
                  : "text-gray-600 hover:bg-[#eae8e2]"
              }`}
            >
              <Files size={16} />
              Documents
            </button>
          </nav>
        </div>

        <div className="mt-6 min-h-0 flex-1 overflow-y-auto px-3">
          <div className="mb-3 flex items-center justify-between px-2">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-gray-400">
              Your chats
            </p>
            {loadingConversations && (
              <LoaderCircle size={13} className="animate-spin text-gray-400" />
            )}
          </div>

          {conversations.length === 0 && !loadingConversations ? (
            <p className="px-2 py-3 text-xs leading-5 text-gray-400">
              {conversationSearch
                ? "No matching conversations."
                : "Your conversations will appear here."}
            </p>
          ) : (
            <div className="space-y-4">
              {Object.entries(groupedConversations).map(([group, items]) => {
                if (!items.length) return null

                return (
                  <div key={group}>
                    <p className="mb-1 px-2 text-[11px] font-medium text-gray-400">
                      {group}
                    </p>

                    <div className="space-y-0.5">
                      {items.map((conversation) => (
                        <div
                          key={conversation.conversation_id}
                          className={`group relative flex items-center rounded-lg ${
                            activeConversationId ===
                            conversation.conversation_id
                              ? "bg-[#eae8e2]"
                              : "hover:bg-[#eeede8]"
                          }`}
                        >
                          {renamingConversationId ===
                          conversation.conversation_id ? (
                            <form
                              className="flex min-w-0 flex-1 items-center gap-1 px-2 py-1"
                              onSubmit={(event) => {
                                event.preventDefault()
                                renameConversation(conversation.conversation_id)
                              }}
                            >
                              <input
                                autoFocus
                                value={renameValue}
                                onChange={(event) =>
                                  setRenameValue(event.target.value)
                                }
                                onKeyDown={(event) => {
                                  if (event.key === "Escape") {
                                    setRenamingConversationId(null)
                                  }
                                }}
                                className="min-w-0 flex-1 rounded border border-[#d8d4ca] bg-white px-2 py-1 text-xs outline-none"
                              />
                              <button
                                type="submit"
                                className="p-1 text-green-700"
                                title="Save"
                              >
                                <Check size={14} />
                              </button>
                              <button
                                type="button"
                                onClick={() => setRenamingConversationId(null)}
                                className="p-1 text-gray-400"
                                title="Cancel"
                              >
                                <X size={14} />
                              </button>
                            </form>
                          ) : (
                            <>
                              <button
                                onClick={() =>
                                  openConversation(conversation.conversation_id)
                                }
                                className="flex min-w-0 flex-1 items-center gap-2.5 px-2.5 py-2 text-left text-[13px]"
                              >
                                <MessageCircle
                                  size={14}
                                  className="shrink-0 text-gray-400"
                                />
                                <span className="truncate">
                                  {conversation.title}
                                </span>
                              </button>

                              <button
                                onClick={() =>
                                  setConversationMenuId(
                                    conversationMenuId ===
                                      conversation.conversation_id
                                      ? null
                                      : conversation.conversation_id,
                                  )
                                }
                                className={`mr-1 rounded p-1 text-gray-400 hover:bg-[#dedbd3] hover:text-gray-700 ${
                                  conversationMenuId ===
                                  conversation.conversation_id
                                    ? "opacity-100"
                                    : "opacity-0 group-hover:opacity-100"
                                }`}
                                title="Conversation options"
                              >
                                <MoreHorizontal size={16} />
                              </button>
                            </>
                          )}

                          {conversationMenuId ===
                            conversation.conversation_id && (
                            <div className="absolute right-1 top-9 z-30 w-40 rounded-lg border border-[#e5e2dc] bg-white p-1 shadow-lg">
                              <button
                                onClick={() => {
                                  setRenameValue(conversation.title)
                                  setRenamingConversationId(
                                    conversation.conversation_id,
                                  )
                                  setConversationMenuId(null)
                                }}
                                className="flex w-full items-center gap-2 rounded px-3 py-2 text-xs hover:bg-[#f5f4f0]"
                              >
                                <Pencil size={14} />
                                Rename
                              </button>
                              <button
                                onClick={() =>
                                  deleteConversation(
                                    conversation.conversation_id,
                                  )
                                }
                                disabled={
                                  deletingConversationId ===
                                  conversation.conversation_id
                                }
                                className="flex w-full items-center gap-2 rounded px-3 py-2 text-xs text-red-600 hover:bg-red-50"
                              >
                                {deletingConversationId ===
                                conversation.conversation_id ? (
                                  <LoaderCircle
                                    size={14}
                                    className="animate-spin"
                                  />
                                ) : (
                                  <Trash2 size={14} />
                                )}
                                Delete
                              </button>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        <div className="shrink-0 border-t border-[#e5e2dc] p-3">
          <button className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm text-gray-600 hover:bg-[#eae8e2]">
            <Settings size={16} />
            Settings
          </button>

          <div className="mt-3 flex items-center gap-3 px-3 py-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#ded6c8] text-sm font-medium">
              L
            </div>
            <div>
              <p className="text-sm font-medium">LexiCore User</p>
              <p className="text-xs text-gray-500">Personal workspace</p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main workspace */}
      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-[#e9e7e2] px-5">
          <div className="flex items-center gap-3">
            {!sidebarOpen && (
              <button
                onClick={() => setSidebarOpen(true)}
                className="rounded-md p-1.5 text-gray-500 hover:bg-[#f0eee9]"
                title="Open sidebar"
              >
                <PanelLeftOpen size={18} />
              </button>
            )}

            <span className="text-sm font-medium">
              {activePage === "chat" ? "Chat" : "Documents"}
            </span>

            {activePage === "chat" && activeConversation && (
              <span className="max-w-[300px] truncate text-sm text-gray-400">
                / {activeConversation.title}
              </span>
            )}
          </div>

          {activePage === "chat" && (
            <button
              onClick={startNewConversation}
              className="flex items-center gap-2 rounded-lg border border-[#e5e2dc] px-3 py-1.5 text-xs text-gray-600 hover:bg-white"
            >
              <Plus size={14} />
              New chat
            </button>
          )}
        </header>

        {/* Chat page */}
        {activePage === "chat" && (
          <section className="flex min-h-0 flex-1 flex-col">
            <div className="flex-1 overflow-y-auto px-5 py-8">
              <div className="mx-auto flex w-full max-w-3xl flex-col">
                {loadingConversation ? (
                  <div className="flex flex-1 items-center justify-center py-24 text-sm text-gray-400">
                    <LoaderCircle
                      size={18}
                      className="mr-2 animate-spin"
                    />
                    Loading conversation...
                  </div>
                ) : messages.length === 0 ? (
                  <div className="flex min-h-[55vh] flex-col items-center justify-center text-center">
                    <div className="mb-6 flex h-12 w-12 items-center justify-center rounded-2xl bg-[#eae8e2]">
                      <MessageSquare
                        size={22}
                        className="text-[#716d63]"
                      />
                    </div>

                    <h2 className="mb-3 text-3xl font-semibold tracking-tight">
                      What can I help you understand?
                    </h2>

                    <p className="max-w-lg text-sm leading-6 text-gray-500">
                      Ask questions about your documents and let LexiCore
                      retrieve the information you need.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-8">
                    {messages.map((message) => (
                      <article key={message.id} className="w-full">
                        <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-[#8a867d]">
                          {message.role === "user" ? "You" : "LexiCore"}
                        </p>

                        <div
                          className={`whitespace-pre-wrap text-[15px] leading-8 ${
                            message.error ? "text-red-700" : "text-[#292824]"
                          }`}
                        >
                          {message.content}
                        </div>

                        {message.role === "assistant" &&
                          message.sources &&
                          message.sources.length > 0 && (
                            <div className="mt-6 border-t border-[#e9e7e2] pt-4">
                              <p className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-gray-400">
                                Sources · {message.sources.length}
                              </p>

                              <div className="space-y-3">
                                {message.sources.map((source) => {
                                  const sourceDocument = documents.find(
                                    (document) =>
                                      document.document_id ===
                                      source.document_id,
                                  )

                                  return (
                                    <div
                                      key={source.chunk_id}
                                      className="rounded-lg border border-[#e9e7e2] bg-white px-4 py-3"
                                    >
                                      <div className="mb-2 flex items-center gap-2 text-xs font-medium text-gray-600">
                                        <FileText
                                          size={14}
                                          className="shrink-0"
                                        />
                                        <span className="truncate">
                                          {sourceDocument?.filename ||
                                            `Document #${source.document_id}`}
                                        </span>
                                        <span className="ml-auto shrink-0 text-gray-400">
                                          Chunk {source.chunk_index}
                                        </span>
                                      </div>

                                      <p className="whitespace-pre-wrap text-xs leading-6 text-gray-500">
                                        {source.content}
                                      </p>
                                    </div>
                                  )
                                })}
                              </div>
                            </div>
                          )}
                      </article>
                    ))}

                    {sendingMessage && (
                      <div className="flex items-center gap-2 py-2 text-sm text-gray-400">
                        <LoaderCircle size={15} className="animate-spin" />
                        LexiCore is thinking...
                      </div>
                    )}
                  </div>
                )}

                <div ref={chatEndRef} />
              </div>
            </div>

            {/* Composer */}
            <div className="shrink-0 px-5 pb-5">
              <div className="mx-auto w-full max-w-3xl">
                <div className="mb-2 flex items-center gap-2">
                  <span className="text-xs font-medium text-gray-500">
                    Search scope
                  </span>

                  <div className="relative">
                    <select
                      value={selectedDocumentId}
                      onChange={(event) =>
                        setSelectedDocumentId(event.target.value)
                      }
                      disabled={sendingMessage}
                      className="max-w-[260px] appearance-none rounded-lg border border-[#e5e2dc] bg-white py-1.5 pl-3 pr-8 text-xs outline-none hover:bg-[#faf9f6]"
                    >
                      <option value="all">All documents</option>

                      {documents.map((document) => (
                        <option
                          key={document.document_id}
                          value={document.document_id}
                          disabled={
                            !["ready", "completed"].includes(
                              document.status.toLowerCase(),
                            )
                          }
                        >
                          {document.filename}
                          {["ready", "completed"].includes(
                            document.status.toLowerCase(),
                          )
                            ? ""
                            : ` (${document.status})`}
                        </option>
                      ))}
                    </select>

                    <ChevronDown
                      size={13}
                      className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-500"
                    />
                  </div>
                </div>

                <div className="rounded-2xl border border-[#dedbd3] bg-white px-4 py-3 shadow-sm focus-within:border-[#c9c4b9]">
                  <textarea
                    ref={textareaRef}
                    value={chatInput}
                    onChange={(event) => setChatInput(event.target.value)}
                    onKeyDown={handleChatKeyDown}
                    placeholder="Ask anything about your documents..."
                    disabled={sendingMessage || loadingConversation}
                    rows={2}
                    className="max-h-40 min-h-[64px] w-full resize-y bg-transparent py-1 text-sm leading-6 outline-none placeholder:text-gray-400 disabled:opacity-60"
                  />

                  <div className="flex items-center justify-between">
                    <span className="text-[11px] text-gray-400">
                      Enter to send · Shift + Enter for a new line
                    </span>

                    <button
                      onClick={handleSendMessage}
                      disabled={
                        !chatInput.trim() ||
                        sendingMessage ||
                        loadingConversation
                      }
                      className="rounded-lg bg-[#292824] p-2.5 text-white hover:bg-[#44433e] disabled:cursor-not-allowed disabled:opacity-40"
                      title="Send message"
                    >
                      {sendingMessage ? (
                        <LoaderCircle size={17} className="animate-spin" />
                      ) : (
                        <ArrowUp size={17} />
                      )}
                    </button>
                  </div>
                </div>

                <p className="mt-3 text-center text-[11px] text-gray-400">
                  LexiCore can make mistakes. Verify important information.
                </p>
              </div>
            </div>
          </section>
        )}

        {/* Documents page */}
        {activePage === "documents" && (
          <section className="flex-1 overflow-y-auto p-6 md:p-8">
            <div className="mx-auto max-w-5xl">
              <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h2 className="text-2xl font-semibold tracking-tight">
                    Your Documents
                  </h2>
                  <p className="mt-2 text-sm text-gray-500">
                    Manage and explore your uploaded documents.
                  </p>
                </div>

                <div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,.txt"
                    className="hidden"
                    onChange={handleUpload}
                  />

                  <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploading}
                    className="flex items-center gap-2 rounded-lg bg-[#292824] px-4 py-2.5 text-sm text-white hover:bg-[#44433e] disabled:opacity-50"
                  >
                    {uploading ? (
                      <LoaderCircle size={17} className="animate-spin" />
                    ) : (
                      <Upload size={17} />
                    )}
                    {uploading ? "Uploading..." : "Upload document"}
                  </button>
                </div>
              </div>

              {uploadError && (
                <div className="mb-5 flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                  <AlertCircle size={17} />
                  {uploadError}
                </div>
              )}

              {documentsError && (
                <div className="mb-5 flex items-center justify-between gap-3 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                  <span>{documentsError}</span>
                  <button
                    onClick={fetchDocuments}
                    className="font-medium underline"
                  >
                    Retry
                  </button>
                </div>
              )}

              <div className="mb-6 flex items-center gap-3 rounded-xl border border-[#e5e2dc] bg-white px-4 py-3">
                <Search size={18} className="text-gray-400" />
                <input
                  value={documentSearch}
                  onChange={(event) =>
                    setDocumentSearch(event.target.value)
                  }
                  placeholder="Search documents..."
                  className="w-full bg-transparent text-sm outline-none placeholder:text-gray-400"
                />
              </div>

              <div className="overflow-visible rounded-xl border border-[#e5e2dc] bg-white">
                <div className="grid grid-cols-12 rounded-t-xl border-b border-[#e9e7e2] bg-[#f9f8f5] px-5 py-3 text-xs font-medium text-gray-500">
                  <span className="col-span-6">DOCUMENT</span>
                  <span className="col-span-3">STATUS</span>
                  <span className="col-span-2">TYPE</span>
                  <span className="col-span-1" />
                </div>

                {filteredDocuments.length > 0 ? (
                  filteredDocuments.map((document) => {
                    const status = getStatusDisplay(document.status)

                    return (
                      <div
                        key={document.document_id}
                        className="grid grid-cols-12 items-center border-b border-[#f0eee9] px-5 py-4 last:border-0"
                      >
                        <div className="col-span-6 flex min-w-0 items-center gap-3">
                          <FileText
                            size={19}
                            className="shrink-0 text-gray-400"
                          />
                          <span className="truncate text-sm font-medium">
                            {document.filename}
                          </span>
                        </div>

                        <div className="col-span-3">
                          <span
                            className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs ${status.className}`}
                          >
                            {status.icon}
                            {status.label}
                          </span>
                        </div>

                        <span className="col-span-2 text-sm uppercase text-gray-500">
                          {document.filename.split(".").pop()}
                        </span>

                        <div className="relative col-span-1 text-right">
                          <button
                            onClick={() =>
                              setDocumentMenuId(
                                documentMenuId === document.document_id
                                  ? null
                                  : document.document_id,
                              )
                            }
                            className="rounded-md p-1.5 text-gray-400 hover:bg-[#f0eee9] hover:text-gray-700"
                            title="Document options"
                          >
                            {deletingDocumentId === document.document_id ? (
                              <LoaderCircle
                                size={17}
                                className="animate-spin"
                              />
                            ) : (
                              <MoreHorizontal size={18} />
                            )}
                          </button>

                          {documentMenuId === document.document_id && (
                            <div className="absolute right-0 top-9 z-20 w-44 rounded-lg border border-[#e5e2dc] bg-white p-1 text-left shadow-lg">
                              <button
                                onClick={() => {
                                  setSelectedDocumentDetails(document)
                                  setDocumentMenuId(null)
                                }}
                                className="flex w-full items-center gap-2 rounded px-3 py-2 text-xs hover:bg-[#f5f4f0]"
                              >
                                <FileText size={14} />
                                View details
                              </button>

                              <button
                                onClick={() => {
                                  copyText(String(document.document_id))
                                  setDocumentMenuId(null)
                                }}
                                className="flex w-full items-center gap-2 rounded px-3 py-2 text-xs hover:bg-[#f5f4f0]"
                              >
                                <Copy size={14} />
                                Copy document ID
                              </button>

                              <button
                                onClick={() => deleteDocument(document)}
                                className="flex w-full items-center gap-2 rounded px-3 py-2 text-xs text-red-600 hover:bg-red-50"
                              >
                                <Trash2 size={14} />
                                Delete document
                              </button>
                            </div>
                          )}
                        </div>

                        {document.status.toLowerCase() === "failed" &&
                          document.last_error && (
                            <p className="col-span-12 mt-2 text-xs text-red-600">
                              {document.last_error}
                            </p>
                          )}
                      </div>
                    )
                  })
                ) : (
                  <div className="flex flex-col items-center justify-center px-5 py-20 text-center">
                    <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-[#f3f1eb]">
                      <FileText size={25} className="text-gray-500" />
                    </div>

                    <h3 className="text-sm font-medium">
                      {documentSearch
                        ? "No matching documents"
                        : "No documents yet"}
                    </h3>

                    <p className="mt-2 max-w-sm text-sm text-gray-500">
                      {documentSearch
                        ? "Try another search term."
                        : "Upload your first document to start organizing, analyzing, and searching your files."}
                    </p>

                    {!documentSearch && (
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="mt-5 flex items-center gap-2 rounded-lg border border-[#e5e2dc] px-4 py-2 text-sm hover:bg-[#f5f4f0]"
                      >
                        <Plus size={16} />
                        Add your first document
                      </button>
                    )}
                  </div>
                )}
              </div>

              <p className="mt-4 text-xs text-gray-400">
                {readyDocuments.length} of {documents.length} documents ready
                for chat.
              </p>
            </div>
          </section>
        )}
      </main>

      {/* Document details modal */}
      {selectedDocumentDetails && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/20 p-4"
          onClick={() => setSelectedDocumentDetails(null)}
        >
          <div
            className="w-full max-w-md rounded-xl border border-[#e5e2dc] bg-white p-6 shadow-xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="mb-5 flex items-center justify-between">
              <h3 className="text-lg font-semibold">Document details</h3>
              <button
                onClick={() => setSelectedDocumentDetails(null)}
                className="rounded-md p-1.5 text-gray-400 hover:bg-gray-100"
              >
                <X size={18} />
              </button>
            </div>

            <div className="space-y-4 text-sm">
              <div>
                <p className="mb-1 text-xs text-gray-400">Filename</p>
                <p className="break-all font-medium">
                  {selectedDocumentDetails.filename}
                </p>
              </div>

              <div>
                <p className="mb-1 text-xs text-gray-400">Document ID</p>
                <div className="flex items-center justify-between gap-2">
                  <span>{selectedDocumentDetails.document_id}</span>
                  <button
                    onClick={() =>
                      copyText(String(selectedDocumentDetails.document_id))
                    }
                    className="rounded-md p-1.5 text-gray-500 hover:bg-gray-100"
                    title="Copy ID"
                  >
                    <Copy size={15} />
                  </button>
                </div>
              </div>

              <div>
                <p className="mb-1 text-xs text-gray-400">Status</p>
                <span>
                  {getStatusDisplay(selectedDocumentDetails.status).label}
                </span>
              </div>

              <div>
                <p className="mb-1 text-xs text-gray-400">Processing attempts</p>
                <p>{selectedDocumentDetails.attempt_count}</p>
              </div>

              {selectedDocumentDetails.last_error && (
                <div>
                  <p className="mb-1 text-xs text-gray-400">Last error</p>
                  <p className="break-words text-red-600">
                    {selectedDocumentDetails.last_error}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Toast */}
      {notice && (
        <div className="fixed bottom-5 right-5 z-[60] flex items-center gap-2 rounded-lg bg-[#292824] px-4 py-3 text-sm text-white shadow-lg">
          <CheckCircle2 size={16} />
          {notice}
        </div>
      )}
    </div>
  )
}

export default App
