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
    categories: [],
    brands: [],
    rating: 0,
    priceMin: 0,
    priceMax: 50000,
    deliveryDays: 5,
  })
  const [sort, setSort] = useState("relevance")
  const [userLat, setUserLat] = useState<number | null>(null)
  const [userLon, setUserLon] = useState<number | null>(null)
  const [showMobileFilters, setShowMobileFilters] = useState(false)

  // Mandatory geolocation on app load
  useEffect(() => {
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setUserLat(pos.coords.latitude)
        setUserLon(pos.coords.longitude)
        console.log("Location obtained:", pos.coords.latitude, pos.coords.longitude)
      },
      (err) => {
        console.warn("Location denied, defaulting to Delhi coordinates")
        // Fallback to Delhi coordinates
        setUserLat(28.66)
        setUserLon(77.23)
      },
    )
  }, [])

  return (
    <div className="min-h-screen bg-gray-50">
      <Header onSearch={setQuery} />
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
              className="flex items-center justify-between w-full text-left"
            >
              <span className="font-medium">Filters</span>
              <span className="text-sm text-gray-500">{showMobileFilters ? "Hide" : "Show"}</span>
            </button>
          </div>

          {/* Filters Sidebar */}
          <div className={`${showMobileFilters ? "block" : "hidden"} lg:block`}>
            <FiltersSidebar filters={filters} onChange={setFilters} />
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
                  className="border rounded px-3 py-1 text-sm w-full sm:w-auto"
                >
                  <option value="relevance">Relevance</option>
                  <option value="price_low_high">Price -- Low to High</option>
                  <option value="price_high_low">Price -- High to Low</option>
                  <option value="rating">Customer Rating</option>
                  <option value="newest">Newest First</option>
                </select>
              </div>
            </div>
            <ResultsGrid query={query} filters={filters} sort={sort} userLat={userLat} userLon={userLon} />
          </div>
        </div>
      )}
    </div>
  )
}
