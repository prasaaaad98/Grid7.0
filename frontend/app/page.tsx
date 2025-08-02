"use client"

import { useState, useEffect } from "react"
import Header from "@/components/Header"
import TrendingBar from "@/components/TrendingBar"
import FiltersSidebar from "@/components/FiltersSidebar"
import ResultsGrid from "@/components/ResultsGrid"
import WelcomeSection from "@/components/WelcomeSection"

export default function App() {
  const [query, setQuery] = useState("")
  const [filters, setFilters] = useState({
    categories: [] as string[],
    brands: [] as string[],
    rating: 0,
    priceMin: 0,
    priceMax: 50000,
    deliveryDays: 5,
  })
  const [sort, setSort] = useState("relevance")
  const [userLat, setUserLat] = useState<number | null>(null)
  const [userLon, setUserLon] = useState<number | null>(null)
  const [showMobileFilters, setShowMobileFilters] = useState(false)
  const [isSearchLoading, setIsSearchLoading] = useState(false)

  const handleHomeClick = () => {
    setQuery("")
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  // Mandatory geolocation on app load
  useEffect(() => {
    console.log("Requesting user location...")
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setUserLat(pos.coords.latitude)
        setUserLon(pos.coords.longitude)
        console.log("✅ Location obtained successfully:", pos.coords.latitude, pos.coords.longitude)
      },
      (err) => {
        console.warn("❌ Location denied, defaulting to Bangalore coordinates:", err)
        // Fallback to Bangalore coordinates (more central location in India)
        setUserLat(12.9716)
        setUserLon(77.5946)
        console.log("🏙️ Using fallback location: Bangalore (12.9716, 77.5946)")
      },
    )
  }, [])

  return (
    <div className="min-h-screen bg-gray-50">
      <Header onSearch={setQuery} onHomeClick={handleHomeClick} />
      <TrendingBar onTrendingClick={setQuery} />

      {!query ? (
        // Welcome section when no search
        <WelcomeSection onCategoryClick={setQuery} />
      ) : (
        // Search results layout with filters
        <div className="flex flex-col lg:flex-row max-w-7xl mx-auto">
          {/* Mobile Filter Toggle */}
          <div className="lg:hidden bg-white border-b p-4">
            <button
              onClick={() => setShowMobileFilters(!showMobileFilters)}
              className="flex items-center justify-between w-full text-left transition-all duration-200 hover:bg-gray-50 p-2 rounded"
            >
              <span className="font-medium">Filters</span>
              <span className={`text-sm text-gray-500 transition-transform duration-200 ${showMobileFilters ? 'rotate-180' : ''}`}>
                {showMobileFilters ? "▲ Hide" : "▼ Show"}
              </span>
            </button>
          </div>

          {/* Filters Sidebar */}
          <div className={`transition-all duration-300 ease-in-out overflow-hidden ${
            showMobileFilters ? "block max-h-screen opacity-100" : "hidden lg:block max-h-0 lg:max-h-screen lg:opacity-100"
          }`}>
            <FiltersSidebar filters={filters} onChange={setFilters} isLoading={isSearchLoading} />
          </div>

          <div className="flex-1">
            <div className="bg-white p-4 border-b">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div className="text-sm text-gray-600">
                  Showing results for "<span className="font-medium">{query}</span>"
                </div>
                <select
                  value={sort}
                  onChange={(e) => setSort(e.target.value)}
                  className="border rounded px-3 py-1 text-sm w-full sm:w-auto transition-all duration-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                >
                  <option value="relevance">Relevance</option>
                  <option value="price_low_high">Price -- Low to High</option>
                  <option value="price_high_low">Price -- High to Low</option>
                  <option value="rating">Customer Rating</option>
                  <option value="newest">Newest First</option>
                </select>
              </div>
            </div>
            <ResultsGrid 
              query={query} 
              filters={filters} 
              sort={sort} 
              userLat={userLat} 
              userLon={userLon} 
              onLoadingChange={setIsSearchLoading}
            />
          </div>
        </div>
      )}
    </div>
  )
}
