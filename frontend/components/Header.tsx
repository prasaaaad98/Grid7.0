"use client"

import React, { useState, useEffect } from 'react'
import { Search, ShoppingCart, User } from "lucide-react"
import LoginModal from "./LoginModal"
import CartModal from "./CartModal"

interface HeaderProps {
  onSearch: (query: string) => void
  onHomeClick?: () => void
}

export default function Header({ onSearch, onHomeClick }: HeaderProps) {
  const [cartCount, setCartCount] = useState(0)
  const [showLogin, setShowLogin] = useState(false)
  const [showCart, setShowCart] = useState(false)
  const [currentUser, setCurrentUser] = useState<any>(null)

  const handleHomeClick = () => {
    // Clear search and go home
    if (onHomeClick) {
      onHomeClick()
    }
    // Force re-render of search component by dispatching custom event
    window.dispatchEvent(new CustomEvent('clearSearch'))
  }

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
            <div 
              className="flex items-center cursor-pointer hover:opacity-80 transition-opacity"
              onClick={handleHomeClick}
            >
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
          <div 
            className="flex items-center cursor-pointer hover:opacity-80 transition-opacity"
            onClick={handleHomeClick}
          >
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
  const [selectedIndex, setSelectedIndex] = useState(-1)
  const [inputRef, setInputRef] = useState<HTMLInputElement | null>(null)

  // Listen for clear search event from home button
  useEffect(() => {
    const handleClearSearch = () => {
      setQuery("")
      setSuggestions([])
      setShowSuggestions(false)
      setSelectedIndex(-1)
    }

    window.addEventListener('clearSearch', handleClearSearch)
    
    return () => {
      window.removeEventListener('clearSearch', handleClearSearch)
    }
  }, [])

  // Global ESC key handler
  useEffect(() => {
    const handleGlobalEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        // Close suggestions and blur input
        setShowSuggestions(false)
        setSelectedIndex(-1)
        setIsFocused(false)
        
        // Blur the input if it's focused
        if (inputRef && document.activeElement === inputRef) {
          inputRef.blur()
        }
      }
    }

    // Add global listener
    document.addEventListener('keydown', handleGlobalEscape)
    
    return () => {
      document.removeEventListener('keydown', handleGlobalEscape)
    }
  }, [inputRef])

  useEffect(() => {
    if (!query.trim()) {
      // Show trending products when search is empty but focused
      if (isFocused) {
        const trendingProducts = [
          "iPhone 15",
          "Samsung Galaxy S24",
          "AirPods Pro",
          "MacBook Air",
          "iPad",
          "Sony WH-1000XM5",
          "Google Pixel 8",
          "OnePlus 12"
        ]
        setSuggestions(trendingProducts)
        setShowSuggestions(true)
      } else {
        setSuggestions([])
        setShowSuggestions(false)
      }
      return
    }

    // Get suggestions from backend when user types
    fetch(`http://localhost:8000/autosuggest?q=${encodeURIComponent(query)}`)
      .then(res => res.json())
      .then(data => {
        setSuggestions(data)
        setShowSuggestions(true)
        setSelectedIndex(-1)
      })
      .catch(error => {
        console.error('Error:', error)
        setSuggestions([])
        setShowSuggestions(false)
      })
  }, [query, isFocused])

  // Updated handleSelect function to handle both cases consistently
  function handleSelect(selectedQuery: string) {
    // Save to recent searches
    const recents = JSON.parse(localStorage.getItem("recentSearches") || "[]")
    localStorage.setItem("recentSearches", JSON.stringify([selectedQuery, ...recents].slice(0, 10)))
    
    // Update query state
    setQuery(selectedQuery)
    
    // Hide suggestions
    setShowSuggestions(false)
    setSelectedIndex(-1)
    setIsFocused(false)
  }

  // Separate function to trigger search
  function triggerSearch(searchQuery: string = query) {
    if (!searchQuery.trim()) return
    
    // Save to recent searches if not already there
    const recents = JSON.parse(localStorage.getItem("recentSearches") || "[]")
    if (!recents.includes(searchQuery)) {
      localStorage.setItem("recentSearches", JSON.stringify([searchQuery, ...recents].slice(0, 10)))
    }
    
    // Trigger the search
    onSearch(searchQuery)
  }

  // Handle mouse click on suggestions
  function handleSuggestionClick(suggestion: string) {
    handleSelect(suggestion)
    // Small delay to ensure state is updated before triggering search
    setTimeout(() => {
      triggerSearch(suggestion)
    }, 0)
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (!showSuggestions || suggestions.length === 0) {
      if (e.key === "Enter") {
        triggerSearch() // Just trigger search with current query
      }
      return
    }

    switch (e.key) {
      case "ArrowDown":
        e.preventDefault()
        setSelectedIndex(prev => 
          prev < suggestions.length - 1 ? prev + 1 : prev
        )
        break
      case "ArrowUp":
        e.preventDefault()
        setSelectedIndex(prev => prev > 0 ? prev - 1 : -1)
        break
      case "Enter":
        e.preventDefault()
        if (selectedIndex >= 0 && selectedIndex < suggestions.length) {
          const selectedSuggestion = suggestions[selectedIndex]
          handleSelect(selectedSuggestion)
          setTimeout(() => {
            triggerSearch(selectedSuggestion)
          }, 0)
        } else {
          triggerSearch() // Search current query when Enter is pressed
        }
        break
      case "Escape":
        e.preventDefault()
        setShowSuggestions(false)
        setSelectedIndex(-1)
        setIsFocused(false)
        // Blur the input field
        if (e.target instanceof HTMLInputElement) {
          e.target.blur()
        }
        break
    }
  }

  return (
    <div className="relative">
      <div className="flex bg-white rounded">
        <input
          ref={(el) => setInputRef(el)}
          type="text"
          placeholder="Search for products, brands and more"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value)
            setSelectedIndex(-1) // Reset selection when typing
          }}
          onKeyDown={handleKeyDown}
          onFocus={() => setIsFocused(true)}
          onBlur={() => {
            // Delay hiding suggestions to allow clicking on them
            setTimeout(() => {
              setIsFocused(false)
              setShowSuggestions(false)
              setSelectedIndex(-1)
            }, 200)
          }}
          className="flex-1 px-4 py-2 text-gray-800 outline-none rounded-l"
        />
        <button onClick={() => triggerSearch()} className="bg-[#2874f0] px-4 py-2 rounded-r hover:bg-blue-600">
          <Search size={20} />
        </button>
      </div>

      {showSuggestions && suggestions.length > 0 && (
        <div className="absolute top-full left-0 bg-white border border-gray-200 rounded-b shadow-lg z-50" style={{ width: 'calc(100% - 48px)' }}>
          {!query.trim() && (
            <div className="px-4 py-2 text-xs text-gray-500 border-b bg-gray-50">
              Trending Searches
            </div>
          )}
          {suggestions.map((suggestion, index) => (
            <div
              key={index}
              onClick={() => handleSuggestionClick(suggestion)}
              className={`px-4 py-2 cursor-pointer text-gray-800 border-b last:border-b-0 ${
                index === selectedIndex 
                  ? 'bg-blue-100 text-blue-800' 
                  : 'hover:bg-gray-100'
              }`}
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