"use client"

import { useState, useEffect } from "react"
import { Search, ShoppingCart, User } from "lucide-react"
import LoginModal from "./LoginModal"
import CartModal from "./CartModal"

interface HeaderProps {
  onSearch: (query: string) => void
}

export default function Header({ onSearch }: HeaderProps) {
  const [cartCount, setCartCount] = useState(0)
  const [showLogin, setShowLogin] = useState(false)
  const [showCart, setShowCart] = useState(false)
  const [currentUser, setCurrentUser] = useState<any>(null)

  useEffect(() => {
    const updateCartCount = () => {
      const cart = JSON.parse(localStorage.getItem("cart") || "[]")
      const totalItems = cart.reduce((total: number, item: any) => total + (item.quantity || 1), 0)
      setCartCount(totalItems)
    }

    const updateUser = () => {
      const user = JSON.parse(localStorage.getItem("currentUser") || "null")
      setCurrentUser(user)
    }

    // Initial load
    updateCartCount()
    updateUser()

    // Listen for storage events
    window.addEventListener("storage", updateCartCount)
    window.addEventListener("storage", updateUser)

    return () => {
      window.removeEventListener("storage", updateCartCount)
      window.removeEventListener("storage", updateUser)
    }
  }, [])

  const handleLogout = () => {
    localStorage.removeItem("currentUser")
    setCurrentUser(null)
    alert("Logged out successfully!")
  }

  return (
    <header className="bg-[#2874f0] text-white sticky top-0 z-50">
      <div className="px-4 py-2">
        {/* Mobile Layout */}
        <div className="md:hidden">
          {/* Top row - Logo and icons */}
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center">
              <div className="text-lg font-bold">Flipkart</div>
              <div className="text-xs text-yellow-300 ml-1">Plus</div>
            </div>

            <div className="flex items-center space-x-4">
              {currentUser ? (
                <div className="flex items-center space-x-2">
                  <User size={18} />
                  <button onClick={handleLogout} className="text-xs">
                    Logout
                  </button>
                </div>
              ) : (
                <button onClick={() => setShowLogin(true)} className="flex items-center space-x-1">
                  <User size={18} />
                  <span className="text-xs">Login</span>
                </button>
              )}

              <button onClick={() => setShowCart(true)} className="relative">
                <ShoppingCart size={18} />
                {cartCount > 0 && (
                  <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-4 h-4 flex items-center justify-center text-[10px]">
                    {cartCount}
                  </span>
                )}
              </button>
            </div>
          </div>

          {/* Search bar - full width on mobile */}
          <div className="w-full">
            <SearchBar onSearch={onSearch} />
          </div>
        </div>

        {/* Desktop Layout */}
        <div className="hidden md:flex items-center justify-between max-w-7xl mx-auto">
          {/* Logo */}
          <div className="flex items-center">
            <div className="text-xl font-bold">Flipkart</div>
            <div className="text-xs text-yellow-300 ml-1">Plus</div>
          </div>

          {/* Search Bar */}
          <div className="flex-1 max-w-2xl mx-8">
            <SearchBar onSearch={onSearch} />
          </div>

          {/* Right side icons */}
          <div className="flex items-center space-x-6">
            {currentUser ? (
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-1">
                  <User size={20} />
                  <span className="text-sm">Hi, {currentUser.name}</span>
                </div>
                <button onClick={handleLogout} className="text-sm hover:bg-blue-600 px-2 py-1 rounded">
                  Logout
                </button>
              </div>
            ) : (
              <div
                className="flex items-center space-x-1 cursor-pointer hover:bg-blue-600 px-2 py-1 rounded"
                onClick={() => setShowLogin(true)}
              >
                <User size={20} />
                <span className="text-sm">Login</span>
              </div>
            )}

            <div
              className="flex items-center space-x-1 cursor-pointer relative hover:bg-blue-600 px-2 py-1 rounded"
              onClick={() => setShowCart(true)}
            >
              <ShoppingCart size={20} />
              <span className="text-sm">Cart</span>
              {cartCount > 0 && (
                <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                  {cartCount}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Login Modal */}
      {showLogin && <LoginModal onClose={() => setShowLogin(false)} />}

      {/* Cart Modal */}
      {showCart && <CartModal onClose={() => setShowCart(false)} />}
    </header>
  )
}

function SearchBar({ onSearch }: { onSearch: (query: string) => void }) {
  const [query, setQuery] = useState("")
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [isFocused, setIsFocused] = useState(false)

  useEffect(() => {
    // Only fetch suggestions if focused or if there's a query
    if (!isFocused && !query.trim()) {
      setSuggestions([])
      setShowSuggestions(false)
      return
    }

    // Real backend call for autosuggest
    fetch(`http://localhost:8000/autosuggest?q=${encodeURIComponent(query)}`)
      .then(res => res.json())
      .then(data => {
        setSuggestions(data)
        setShowSuggestions(true)
      })
      .catch(error => {
        console.error('Error fetching suggestions:', error)
        // No fallback - just show empty suggestions if backend fails
        setSuggestions([])
        setShowSuggestions(false)
      })
  }, [query, isFocused])

  function handleSelect(selectedQuery: string) {
    const recents = JSON.parse(localStorage.getItem("recentSearches") || "[]")
    localStorage.setItem("recentSearches", JSON.stringify([selectedQuery, ...recents].slice(0, 10)))
    setQuery(selectedQuery)
    setShowSuggestions(false)
    onSearch(selectedQuery)
  }

  return (
    <div className="relative">
      <div className="flex bg-white rounded">
        <input
          type="text"
          placeholder="Search for products, brands and more"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              handleSelect(query)
            }
          }}
          onFocus={() => setIsFocused(true)}
          onBlur={() => {
            // Delay hiding suggestions to allow clicking on them
            setTimeout(() => {
              setIsFocused(false)
              setShowSuggestions(false)
            }, 200)
          }}
          className="flex-1 px-4 py-2 text-gray-800 outline-none rounded-l"
        />
        <button onClick={() => handleSelect(query)} className="bg-[#2874f0] px-4 py-2 rounded-r hover:bg-blue-600">
          <Search size={20} />
        </button>
      </div>

      {showSuggestions && suggestions.length > 0 && (
        <div className="absolute top-full left-0 bg-white border border-gray-200 rounded-b shadow-lg z-50" style={{ width: 'calc(100% - 48px)' }}>
          {suggestions.map((suggestion, index) => (
            <div
              key={index}
              onClick={() => handleSelect(suggestion)}
              className="px-4 py-2 hover:bg-gray-100 cursor-pointer text-gray-800 border-b last:border-b-0"
            >
              <Search size={16} className="inline mr-2 text-gray-400" />
              {suggestion}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
